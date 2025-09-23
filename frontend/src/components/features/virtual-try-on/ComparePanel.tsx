import React, { useCallback, useMemo, useState } from 'react';
import { ImageUploader } from './ImageUploader';
import { Card, Button, Spinner, toast, useToast } from '../../ui';
import type { UploadedImage, ApiFile, ClothingItems } from '../../../types';
import { virtualTryOnService } from '../../../services/virtualTryOn.service';
import { shareOrDownloadResult } from '../../../utils/shareImage';

const featureEnabled = (): boolean => {
  const v = (import.meta as any).env?.VITE_FEATURE_COMPARE;
  if (v === undefined || v === null) return true; // default ON
  const s = String(v).toLowerCase();
  return !(s === '0' || s === 'false' || s === 'off');
};

function toApi(img: UploadedImage | null | undefined): ApiFile | null {
  return img ? { base64: img.base64, mimeType: img.mimeType } : null;
}

export const ComparePanel: React.FC<{ basePerson?: UploadedImage | null }>
  = ({ basePerson }) => {
  const { addToast } = useToast();
  const [aTop, setATop] = useState<UploadedImage | null>(null);
  const [aPants, setAPants] = useState<UploadedImage | null>(null);
  const [aShoes, setAShoes] = useState<UploadedImage | null>(null);
  const [aOut, setAOut] = useState<string | null>(null);
  const [aLoading, setALoading] = useState(false);

  const [bTop, setBTop] = useState<UploadedImage | null>(null);
  const [bPants, setBPants] = useState<UploadedImage | null>(null);
  const [bShoes, setBShoes] = useState<UploadedImage | null>(null);
  const [bOut, setBOut] = useState<string | null>(null);
  const [bLoading, setBLoading] = useState(false);

  const canShow = featureEnabled();
  const canGenA = !!basePerson && (aTop || aPants || aShoes);
  const canGenB = !!basePerson && (bTop || bPants || bShoes);

  const generate = useCallback(async (which: 'A'|'B') => {
    const isA = which === 'A';
    const top = isA ? aTop : bTop;
    const pants = isA ? aPants : bPants;
    const shoes = isA ? aShoes : bShoes;
    if (!basePerson || !(top || pants || shoes)) {
      addToast(toast.info('인물 이미지와 최소 1개의 의류가 필요합니다.'));
      return;
    }
    try {
      isA ? setALoading(true) : setBLoading(true);
      const payload: { person: ApiFile; clothingItems: ClothingItems } = {
        person: { base64: basePerson.base64, mimeType: basePerson.mimeType },
        clothingItems: { top: toApi(top), pants: toApi(pants), shoes: toApi(shoes) },
      } as any;
      const res = await virtualTryOnService.combineImages(payload as any);
      (isA ? setAOut : setBOut)(res.generatedImage || null);
    } catch (e: any) {
      addToast(toast.error('생성 실패', e?.message));
    } finally {
      isA ? setALoading(false) : setBLoading(false);
    }
  }, [basePerson, aTop, aPants, aShoes, bTop, bPants, bShoes, addToast]);

  if (!canShow) return null;

  return (
    <Card className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-gray-800">코디 비교 모드 (A/B)</h3>
        {!basePerson && <span className="text-sm text-gray-500">좌측 패널에서 인물 이미지를 먼저 선택하세요.</span>}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Panel A */}
        <div className="space-y-3" id="compare-panel-a">
          <h4 className="font-semibold text-gray-700">A</h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <ImageUploader id="cmp-a-top" title="Top" description="상의" onImageUpload={setATop} externalImage={aTop} />
            <ImageUploader id="cmp-a-pants" title="Pants" description="하의" onImageUpload={setAPants} externalImage={aPants} />
            <ImageUploader id="cmp-a-shoes" title="Shoes" description="신발" onImageUpload={setAShoes} externalImage={aShoes} />
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={() => generate('A')} disabled={!canGenA} loading={aLoading}>Generate A</Button>
            <Button variant="outline" disabled={!aOut} onClick={() => shareOrDownloadResult('#compare-panel-a', aOut || undefined, 'compare-A.png')}>Share</Button>
          </div>
          <div className="bg-gray-50 rounded-lg border border-gray-200 min-h-[160px] flex items-center justify-center p-2">
            {aLoading ? (<Spinner />) : (aOut ? (<img src={aOut} alt="A" className="max-h-64 object-contain" />) : (<span className="text-sm text-gray-500">A 결과가 아직 없습니다</span>))}
          </div>
        </div>
        {/* Panel B */}
        <div className="space-y-3" id="compare-panel-b">
          <h4 className="font-semibold text-gray-700">B</h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <ImageUploader id="cmp-b-top" title="Top" description="상의" onImageUpload={setBTop} externalImage={bTop} />
            <ImageUploader id="cmp-b-pants" title="Pants" description="하의" onImageUpload={setBPants} externalImage={bPants} />
            <ImageUploader id="cmp-b-shoes" title="Shoes" description="신발" onImageUpload={setBShoes} externalImage={bShoes} />
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={() => generate('B')} disabled={!canGenB} loading={bLoading}>Generate B</Button>
            <Button variant="outline" disabled={!bOut} onClick={() => shareOrDownloadResult('#compare-panel-b', bOut || undefined, 'compare-B.png')}>Share</Button>
          </div>
          <div className="bg-gray-50 rounded-lg border border-gray-200 min-h-[160px] flex items-center justify-center p-2">
            {bLoading ? (<Spinner />) : (bOut ? (<img src={bOut} alt="B" className="max-h-64 object-contain" />) : (<span className="text-sm text-gray-500">B 결과가 아직 없습니다</span>))}
          </div>
        </div>
      </div>
    </Card>
  );
};

export default ComparePanel;

