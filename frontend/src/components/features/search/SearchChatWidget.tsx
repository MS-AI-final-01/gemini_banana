import React, { useMemo, useRef, useState } from 'react';
import { apiClient } from '../../../services/api.service';
import type { RecommendationItem } from '../../../types';

interface Props {
  onApplyResults?: (items: RecommendationItem[], query?: string) => void;
}

type Msg = { role: 'assistant' | 'user'; content: string };

const panelStyle: React.CSSProperties = {
  position: 'fixed',
  right: 24,
  bottom: 24,
  zIndex: 50,
};

const hintStyle: React.CSSProperties = {
  position: 'absolute',
  right: 46,
  bottom: 70,
  background: '#fff',
  color: '#111',
  padding: '6px 12px',
  borderRadius: 999,
  fontSize: 12,
  lineHeight: 1.4,
  whiteSpace: 'nowrap',
  boxShadow: '0 6px 16px rgba(0,0,0,0.15)'
};

const cardStyle: React.CSSProperties = {
  width: 'min(520px, 92vw)',
  height: 'min(82vh, 960px)',
  background: '#fff',
  borderRadius: 16,
  boxShadow: '0 10px 30px rgba(0,0,0,0.18)',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  border: '1px solid rgba(0,0,0,0.08)'
};

const headerStyle: React.CSSProperties = {
  padding: '12px 14px',
  borderBottom: '1px solid rgba(0,0,0,0.06)',
  fontWeight: 700,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between'
};

const bubbleBase: React.CSSProperties = {
  maxWidth: '80%',
  padding: '10px 12px',
  borderRadius: 12,
  fontSize: 14,
  lineHeight: 1.45,
  wordBreak: 'break-word',
};

export const SearchChatWidget: React.FC<Props> = ({ onApplyResults }) => {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([
    { role: 'assistant', content: '어떤 옷을 찾으세요?' }
  ]);
  const listRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !busy, [input, busy]);

  const scrollToEnd = () => {
    requestAnimationFrame(() => {
      try { listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' }); } catch {}
    });
  };

  const handleSend = async () => {
    if (!canSend) return;
    const text = input.trim();
    setInput('');
    setMessages((m) => [...m, { role: 'user', content: text }]);
    setBusy(true);
    scrollToEnd();

    try {
      const parsed: any = await apiClient.post('/api/search/parse', { text });
      const tokens: string[] = Array.isArray(parsed?.tokens) ? parsed.tokens : [];
      const q = [...tokens, ...(Array.isArray(parsed?.colors) ? parsed.colors : [])].join(' ');
      const params: any = { q, limit: '24' };
      if (parsed?.category) params.category = parsed.category;
      if (parsed?.priceRange?.min) params.minPrice = String(parsed.priceRange.min);
      if (parsed?.priceRange?.max) params.maxPrice = String(parsed.priceRange.max);

      const qs = new URLSearchParams(params).toString();
      const items = await apiClient.get<RecommendationItem[]>(`/api/search/semantic?${qs}`);

      // Build assistant response text
      const parts: string[] = [];
      if (parsed?.category) parts.push(`카테고리: ${parsed.category}`);
      if (Array.isArray(parsed?.colors) && parsed.colors.length) parts.push(`색상: ${parsed.colors.join(', ')}`);
      if (parsed?.priceRange?.min || parsed?.priceRange?.max) {
        const min = parsed?.priceRange?.min ? `${parsed.priceRange.min.toLocaleString()}원` : '';
        const max = parsed?.priceRange?.max ? `${parsed.priceRange.max.toLocaleString()}원` : '';
        parts.push(`가격: ${min}${min && max ? ' ~ ' : ''}${max}`);
      }
      const summary = parts.length ? `검색 요약 · ${parts.join(' · ')}` : '조건 없이 검색했어요.';

      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `${summary}\n총 ${items.length}개를 찾았어요.` },
      ]);
      onApplyResults?.(items, q);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: '검색에 실패했어요. 네트워크 상태를 확인해주세요.' },
      ]);
    } finally {
      setBusy(false);
      scrollToEnd();
    }
  };

  // Allow external trigger: window.dispatchEvent(new CustomEvent('open-search-chat'))
  React.useEffect(() => {
    const handler = () => setOpen(true);
    window.addEventListener('open-search-chat', handler as any);
    return () => window.removeEventListener('open-search-chat', handler as any);
  }, []);

  return (
    <div style={panelStyle}>
      {!open && (
        <>
          <div style={hintStyle}>
            찾고 싶은 상품을 채팅형식으로 찾아보세요
          </div>
          <button
            type="button"
            aria-label="검색 챗 열기"
            onClick={() => setOpen(true)}
            style={{
              width: 56,
              height: 56,
              borderRadius: 28,
              background: '#111',
              color: '#fff',
              border: 'none',
              boxShadow: '0 8px 20px rgba(0,0,0,0.22)',
              cursor: 'pointer'
            }}
          >
            💬
          </button>
        </>
      )}

      {open && (
        <div style={cardStyle}>
          <div style={headerStyle}>
            <span>Search Assistant</span>
            <button
              aria-label="닫기"
              onClick={() => setOpen(false)}
              style={{ border: 'none', background: 'transparent', fontSize: 18, cursor: 'pointer' }}
            >
              ×
            </button>
          </div>
          <div ref={listRef} style={{ flex: 1, padding: 12, overflowY: 'auto', background: '#fafafa' }}>
            {messages.map((msg, idx) => (
              <div key={idx} style={{ display: 'flex', margin: '8px 0', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                <div
                  style={{
                    ...bubbleBase,
                    background: msg.role === 'user' ? '#111' : '#fff',
                    color: msg.role === 'user' ? '#fff' : '#111',
                    border: msg.role === 'assistant' ? '1px solid rgba(0,0,0,0.08)' : 'none',
                  }}
                >
                  {msg.content}
                </div>
              </div>
            ))}
          </div>
          <div style={{ padding: 10, borderTop: '1px solid rgba(0,0,0,0.06)', display: 'flex', gap: 8 }}>
            <input
              value={input}
              disabled={busy}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSend(); }}
              placeholder="어떤 옷을 찾으세요? 예: 네이비 와이드 슬랙스 5만원 이하"
              style={{ flex: 1, height: 38, borderRadius: 20, border: '1px solid #ddd', padding: '0 12px', outline: 'none' }}
            />
            <button
              type="button"
              disabled={!canSend}
              onClick={handleSend}
              style={{
                width: 38,
                height: 38,
                borderRadius: 19,
                border: 'none',
                background: canSend ? '#111' : '#888',
                color: '#fff',
                cursor: canSend ? 'pointer' : 'default'
              }}
              aria-label="보내기"
              title="보내기"
            >
              ➤
            </button>
          </div>
          <div style={{ textAlign: 'center', fontSize: 11, color: '#777', paddingBottom: 8 }}>
            AI 추천은 참고용이에요
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchChatWidget;
