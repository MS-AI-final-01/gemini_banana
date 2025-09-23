import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from fastapi import HTTPException
from urllib.parse import quote
from google.auth import default
from google.auth.transport.requests import Request


logger = logging.getLogger(__name__)


def _get_env(name: str, default_value: Optional[str] = None) -> str:
    value = os.getenv(name, default_value)
    if not value:
        raise HTTPException(status_code=500, detail=f"Missing required environment variable: {name}")
    return value


def _extract_base64(image_data: str, fallback_mime: str = "image/png") -> Tuple[str, str]:
    if image_data.startswith("data:"):
        try:
            header, b64 = image_data.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            return b64.strip(), mime
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Invalid data URI for imageData") from exc
    return image_data.strip(), fallback_mime


class VertexVideoService:
    def __init__(self) -> None:
        self._scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        self._token_lock = threading.Lock()
        self._token_cache: Optional[Tuple[str, float]] = None

    def _get_access_token(self) -> str:
        now = time.time()
        with self._token_lock:
            if self._token_cache and now < self._token_cache[1] - 60:
                return self._token_cache[0]

            try:
                # 🆕 로컬과 CI/CD 모두 지원하는 인증 방식
                credentials = self._get_credentials()
            except Exception as exc:
                logger.exception("Failed to load Google credentials")
                raise HTTPException(status_code=503, detail="Unable to load Google credentials") from exc

            if not credentials.valid or not getattr(credentials, "token", None):
                try:
                    credentials.refresh(Request())
                except Exception as exc:
                    logger.exception("Failed to refresh Google credentials")
                    raise HTTPException(status_code=503, detail="Unable to refresh Google credentials") from exc

            token = getattr(credentials, "token", None)
            if not token:
                raise HTTPException(status_code=503, detail="Missing access token from Google credentials")

            expiry = getattr(credentials, "expiry", None)
            expiry_ts = now + 300
            if expiry is not None:
                try:
                    expiry_ts = expiry.timestamp()
                except Exception:
                    pass

            self._token_cache = (token, expiry_ts)
            return token

    def _get_credentials(self):
        """로컬과 CI/CD 모두 호환되는 Google Cloud 인증 (Base64 지원)"""
        import json
        import base64
        from google.oauth2 import service_account
        from google.auth import default

        # 1. Base64로 인코딩된 JSON 읽기 (CI/CD용 - 안전)
        credentials_b64 = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_B64')
        if credentials_b64:
            try:
                # Base64 디코딩 후 JSON 파싱
                credentials_json = base64.b64decode(credentials_b64).decode('utf-8')
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=self._scopes
                )
                logger.info("✅ Google credentials loaded from Base64 environment variable")
                return credentials
            except Exception as e:
                logger.warning(f"Failed to load credentials from Base64 env var: {e}")

        # 2. 일반 JSON 환경 변수 (기존 방식)
        credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        if credentials_json:
            try:
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=self._scopes
                )
                logger.info("✅ Google credentials loaded from JSON environment variable")
                return credentials
            except Exception as e:
                logger.warning(f"Failed to load credentials from JSON env var: {e}")

        # 3. 기본 방식 (로컬 개발용 - 파일 경로)
        try:
            credentials, _ = default(scopes=self._scopes)
            logger.info("✅ Google credentials loaded via default method")
            return credentials
        except Exception as e:
            logger.error(f"Failed to load credentials via default method: {e}")
            raise

    @staticmethod
    def _sanitize_parameters(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not parameters:
            return {}
        safe: Dict[str, Any] = {}
        for key, value in parameters.items():
            if value is None:
                continue
            # Preserve primitive types; avoid coercing numbers/bools to strings
            if isinstance(value, (str, bool, int, float)):
                safe[key] = value
            else:
                safe[key] = str(value)
        return safe

    def _build_payload_variants(
        self,
        *,
        prompt: str,
        image_data: Optional[str],
        mime_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params = self._sanitize_parameters(parameters)
        variants: List[Dict[str, Any]] = []

        # Variant A: prompt string + single image field (bytesBase64Encoded)
        if image_data:
            variants.append(
                {
                    "instances": [
                        {
                            "prompt": prompt,
                            "image": {
                                "mimeType": mime_type,
                                "bytesBase64Encoded": image_data,
                            },
                        }
                    ],
                    "parameters": params,
                }
            )

        # Variant B: nested prompt object with images[] (legacy attempt)
        if image_data:
            variants.append(
                {
                    "instances": [
                        {
                            "prompt": {
                                "text": prompt,
                                "images": [
                                    {"mimeType": mime_type, "imageBytes": image_data}
                                ],
                            }
                        }
                    ],
                    "parameters": params,
                }
            )

        # Variant C: text-only prompt
        variants.append(
            {
                "instances": [
                    {"prompt": prompt}
                ],
                "parameters": params,
            }
        )
        return variants


    def _gcs_media_url(self, gcs_uri: str) -> str:
        if not isinstance(gcs_uri, str) or not gcs_uri.startswith('gs://'):
            raise HTTPException(status_code=400, detail='gcs uri must start with gs://')
        without = gcs_uri[5:]
        if '/' not in without:
            raise HTTPException(status_code=400, detail='invalid gcs uri: missing object path')
        bucket, obj = without.split('/', 1)
        from urllib.parse import quote as _quote
        return f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/{_quote(obj, safe="")}\?alt=media'

    def open_uri_stream(self, uri: str) -> httpx.Response:
        token = self._get_access_token()
        headers = { 'Authorization': f'Bearer {token}' }
        url = self._gcs_media_url(uri) if isinstance(uri, str) and uri.startswith('gs://') else uri
        resp = httpx.get(url, headers=headers, timeout=httpx.Timeout(60.0, connect=10.0))
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:  # noqa: BLE001
            body = exc.response.text if exc.response is not None else str(exc)
            logger.error('Media fetch failed (%s): %s', exc.response.status_code if exc.response else 0, body)
            raise HTTPException(status_code=exc.response.status_code if exc.response else 502, detail=body)
        return resp

    def _post_with_retry(
        self,
        *,
        url: str,
        payload: Dict[str, Any],
        token: str,
        timeout: float,
        max_retries: int = 2,
    ) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= max_retries:
            try:
                resp = httpx.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
                )
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:  # noqa: BLE001
                status_code = exc.response.status_code if exc.response is not None else 502
                body = exc.response.text if exc.response is not None else str(exc)
                logger.error("Vertex request failed (%s): %s", status_code, body)
                is_retryable = status_code in {408, 429, 500, 502, 503, 504}
                if not is_retryable or attempt == max_retries:
                    raise HTTPException(status_code=status_code, detail=body)
                last_exc = exc
            except httpx.HTTPError as exc:  # noqa: BLE001
                logger.warning("Vertex transport error: %s", exc)
                if attempt == max_retries:
                    raise HTTPException(status_code=502, detail=str(exc))
                last_exc = exc
            attempt += 1
            sleep_for = min(2.0 * attempt, 5.0)
            if last_exc:
                logger.warning("Vertex video request retry %s/%s due to %s", attempt, max_retries, last_exc)
            time.sleep(sleep_for)

        raise HTTPException(status_code=502, detail="Vertex AI request failed after retries")

    def start_generation(
        self,
        *,
        prompt: str,
        image_data: str,
        mime_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        project_id = _get_env("VERTEX_PROJECT_ID")
        location_id = _get_env("VERTEX_LOCATION", "us-central1")
        model_id = _get_env("VERTEX_MODEL_ID", "veo-3.0-generate-001")
        api_endpoint = os.getenv("VERTEX_API_ENDPOINT", f"{location_id}-aiplatform.googleapis.com")

        token = self._get_access_token()
        url = (
            f"https://{api_endpoint}/v1/projects/{project_id}/locations/{location_id}/"
            f"publishers/google/models/{model_id}:predictLongRunning"
        )

        last_error: Optional[HTTPException] = None
        for payload in self._build_payload_variants(
            prompt=prompt,
            image_data=image_data,
            mime_type=mime_type,
            parameters=parameters,
        ):
            try:
                resp = self._post_with_retry(url=url, payload=payload, token=token, timeout=120.0)
                return resp.json()
            except HTTPException as exc:
                if 400 <= exc.status_code < 500:
                    last_error = exc
                    continue
                raise
        if last_error:
            raise last_error
        raise HTTPException(status_code=502, detail="Vertex request failed for all payload variants")

    def fetch_operation(self, *, operation_name: str) -> Dict[str, Any]:
        project_id = _get_env("VERTEX_PROJECT_ID")
        location_id = _get_env("VERTEX_LOCATION", "us-central1")
        model_id = _get_env("VERTEX_MODEL_ID", "veo-3.0-fast-generate-001")
        api_endpoint = os.getenv("VERTEX_API_ENDPOINT", f"{location_id}-aiplatform.googleapis.com")

        payload = {"operationName": operation_name}
        token = self._get_access_token()
        url = (
            f"https://{api_endpoint}/v1/projects/{project_id}/locations/{location_id}/"
            f"publishers/google/models/{model_id}:fetchPredictOperation"
        )

        resp = self._post_with_retry(url=url, payload=payload, token=token, timeout=30.0)
        return resp.json()

    @staticmethod
    def collect_video_uris(operation: Dict[str, Any]) -> List[str]:
        uris: List[str] = []
        op = operation.get("operation", operation) if isinstance(operation, dict) else {}
        response = op.get("response", {}) if isinstance(op, dict) else {}
        predictions = response.get("predictions") or (op.get("predictions") if isinstance(op, dict) else [])
        if not isinstance(predictions, list):
            return uris
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            candidates = [
                pred.get("videoUri"),
                pred.get("video_uri"),
                pred.get("outputUri"),
                pred.get("output_uri"),
            ]
            for val in candidates:
                if isinstance(val, str) and val:
                    uris.append(val)
            for key in ("videoUris", "video_uris"):
                val = pred.get(key)
                if isinstance(val, list):
                    uris.extend([v for v in val if isinstance(v, str)])
            for key in ("videos", "generatedVideos"):
                val = pred.get(key)
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            for k in ("uri", "gcsUri", "videoUri"):
                                maybe = item.get(k)
                                if isinstance(maybe, str) and maybe:
                                    uris.append(maybe)

        if isinstance(response, dict):
            content = response.get('content') or []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        media = item.get('media') or item.get('video')
                        if isinstance(media, dict):
                            for k in ('uri','gcsUri','videoUri'):
                                v = media.get(k)
                                if isinstance(v, str) and v:
                                    uris.append(v)
            for k in ('videoUri','video_uri','outputUri','output_uri'):
                v = response.get(k)
                if isinstance(v, str) and v:
                    uris.append(v)
                # Official schema: response.generatedSamples[].video.uri
        if isinstance(response, dict):
            samples = response.get("generatedSamples") or response.get("generated_samples") or []
            if isinstance(samples, list):
                for s in samples:
                    if isinstance(s, dict):
                        vid = s.get("video") or s.get("Video") or s.get("media")
                        if isinstance(vid, dict):
                            for k in ("uri", "gcsUri", "videoUri"):
                                v = vid.get(k)
                                if isinstance(v, str) and v:
                                    uris.append(v)
        return list(dict.fromkeys(uris))


vertex_video_service = VertexVideoService()

