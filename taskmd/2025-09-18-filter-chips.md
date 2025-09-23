# 홈 화면 필터 칩(Filter Chips) 정리

목표: 홈(쇼핑/추천) 섹션에서 간단한 토글형 필터 UI(알약 버튼)를 통해 목록을 빠르게 좁히는 패턴을 사용한다. 현재는 스타일만 바뀌는 더미 상태로, 눌림/선택 감지만 되고 데이터는 변경되지 않는다. 이 문서는 현황과 연결 방식을 기록한다.

## 현재 구조
- 컴포넌트: `frontend/src/components/features/home/FilterChips.tsx`
  - 내부 `chips` 배열을 순회 렌더링
  - 클릭 시 로컬 상태 `active` 갱신 → 선택 스타일(검정 배경)만 변경
  - 상위와의 통신(콜백/상태 공유) 없음
- 사용처: `frontend/src/components/features/ecommerce/ECommerceUI.tsx`
  - 홈 섹션 UI에서 카테고리 행과 같이 노출

## 개선 방향(제안)
1) 상위로 선택 이벤트 전달
   - `FilterChips` 시그니처를 변경해 선택 변경 콜백을 노출
   - 예) `export const FilterChips: React.FC<{ onChange?: (label: string) => void }>`
   - 클릭 시 `onChange?.(chip)` 호출

2) 카테고리/태그 매핑 테이블 도입
   - 칩 라벨 → 내부 검색 파라미터를 매핑(예: 상의→`top`, 아우터→`outer`)
   - 예) `{ label: '미니멀', tags: ['minimal', 'clean'] }`, `{ label: '아우터', category: 'outer' }`

3) 추천 API 연동(간단 경로)
   - 임시: 랜덤 추천 API 사용 → `/api/recommend/random?limit=12&category=<mapped>`
   - 정식: `/api/recommend`에 `options`(minPrice, maxPrice, excludeTags 등) 포함하여 호출

4) 상태 보존(선택 사항)
   - 선택된 칩을 `localStorage` 또는 URL 쿼리스트링으로 유지해 새로고침에도 복구
   - 키 예시: `app:home:filter:active=v1`

## 간단 구현 예시(콜백 추가)
```tsx
// FilterChips.tsx
export const FilterChips: React.FC<{ onChange?: (label: string) => void }>
  = ({ onChange }) => {
  const [active, setActive] = useState(0);
  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((chip, index) => (
        <button
          key={chip}
          onClick={() => { setActive(index); onChange?.(chip); }}
          className={`text-xs rounded-full px-4 py-1.5 border transition-colors ${
            active === index ? 'bg-[#111111] text-white border-[#111111]' :
            'bg-white text-[var(--text-muted)] border-[var(--divider)] hover:text-[var(--text-strong)]'}`}
        >{chip}</button>
      ))}
    </div>
  );
}
```

```tsx
// 상위(ECommerceUI 등)
const handleChip = async (label: string) => {
  const map: Record<string,string> = {
    '상의': 'top', '하의': 'pants', '아우터': 'outer', '신발': 'shoes'
  };
  const cat = map[label];
  if (!cat) return;
  const recs = await apiClient.get(`/api/recommend/random?limit=12&category=${cat}`).catch(() => []);
  setRecommendations(recs);
};

<FilterChips onChange={handleChip} />
```

## 수용 기준(초안)
- 칩 클릭 시 시각적 활성화 + 상위 콜백 1회 호출
- 카테고리형 칩은 `/api/recommend/random`을 통해 최소 8~12개 아이템을 교체 렌더링
- 새로고침 후에도 마지막 칩 선택이 복구(선택 시) 또는 기본 상태 유지(선택 안 함)

## 주의/할 일
- 일부 한글 라벨이 깨져 있는 파일이 있음 → i18n/문자 인코딩 정리 필요(특히 `FilterChips.tsx`)
- 칩 다중 선택, 정렬/가격대 칩 등은 2차 과제로 분리
- 호출 빈도 제한(디바운스/중복 요청 취소) 적용 고려

## 참고 파일
- `frontend/src/components/features/home/FilterChips.tsx`
- `frontend/src/components/features/home/CategoryRow.tsx` (동일한 알약 버튼 패턴 반영)
- `frontend/src/components/features/ecommerce/ECommerceUI.tsx`

