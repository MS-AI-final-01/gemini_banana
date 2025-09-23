import React from 'react';

export const HeroBanner: React.FC = () => {
  return (
    <div className="relative w-full aspect-[16/6] rounded-2xl overflow-hidden bg-gradient-to-r from-gray-900 via-gray-700 to-gray-900 text-white">
      <div className="absolute inset-0 opacity-30 bg-[radial-gradient(circle_at_30%_20%,#ffffff33,transparent_50%)]" />
      <div className="absolute inset-0 p-6 sm:p-10 flex flex-col justify-end">
        <p className="text-sm opacity-80">15% 荑좏룿</p>
        <h2 className="text-2xl sm:text-3xl font-extrabold leading-tight">SEEN × EASTPAK 협업 한정 발매</h2>
      </div>
      <div className="absolute right-4 bottom-3 text-xs bg-white/20 px-2 py-1 rounded-full">1 / 3</div>
    </div>
  );
};

export default HeroBanner;


