import { Link, useLocation } from 'react-router-dom';
import { motion } from 'motion/react';
import { useEffect, useState } from 'react';

export default function Header() {
  const location = useLocation();
  const isPaper = location.pathname === '/paper';
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? 'bg-black/80 backdrop-blur-md border-b border-white/10' : 'bg-transparent'}`}>
      <div className="max-w-[1400px] mx-auto px-[clamp(1.5rem,5vw,4rem)] h-20 flex items-center justify-between">
        <div></div>

        <nav className="hidden md:flex items-center space-x-8">
          <Link 
            to="/" 
            className={`font-mono text-xs tracking-[0.15em] uppercase hover:text-[#00ff88] transition-colors ${!isPaper ? 'text-white font-bold' : 'text-[#888]'}`}
          >
            Home
          </Link>
          <a 
            href="/#architecture" 
            className="font-mono text-xs tracking-[0.15em] uppercase text-[#888] hover:text-[#00ff88] transition-colors"
          >
            Pipeline
          </a>
          <a 
            href="/#dashboard" 
            className="font-mono text-xs tracking-[0.15em] uppercase text-[#888] hover:text-[#00ff88] transition-colors"
          >
            Dashboard
          </a>
          <Link 
            to="/paper" 
            className={`font-mono text-xs tracking-[0.15em] uppercase hover:text-[#00ff88] transition-colors ${isPaper ? 'text-white font-bold' : 'text-[#888]'}`}
          >
            Paper
          </Link>
          <a 
            href="https://github.com/AhmedAlharbii/SDC2026_KAU_AE_TEAM" 
            target="_blank" 
            rel="noopener noreferrer"
            className="font-mono text-xs tracking-[0.15em] uppercase text-white bg-white/10 hover:bg-white/20 px-4 py-2 rounded-sm transition-colors border border-white/10"
          >
            GitHub
          </a>
        </nav>

        {/* Mobile menu button could go here, omitting for simplicity unless requested */}
      </div>
    </header>
  );
}
