import React, { useState } from 'react';
import { ShirtIcon } from '../../icons/ShirtIcon';

export const Header: React.FC = () => {
    const logoUrl = (import.meta as any).env?.VITE_LOGO_URL || '/logo.jpg';
    const [logoOk, setLogoOk] = useState(true);
    return (
        <header className="text-center bg-[var(--nav-bg)]">
            <div className="inline-flex items-center gap-3 bg-white px-4 py-2 rounded-xl shadow-sm border border-gray-200">
                {logoOk ? (
                    <img src={logoUrl} alt="logo" className="w-9 h-9 object-contain" onError={() => setLogoOk(false)} />
                ) : (
                    <ShirtIcon className="w-8 h-8 text-primary-600" />
                )}
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 tracking-tight">
                    로고 적용 데모
                </h1>
            </div>
            <p className="mt-3 max-w-2xl mx-auto text-md text-gray-600">
                상단 로고는 public/logo.jpg(또는 VITE_LOGO_URL) 경로를 사용합니다.
            </p>
        </header>
    );
};
