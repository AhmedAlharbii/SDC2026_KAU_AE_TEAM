import { motion } from 'motion/react';
import { Link } from 'react-router-dom';

export default function Hero() {
  return (
    <section className="relative min-h-[95vh] flex items-center pt-[clamp(4rem,10vw,8rem)] pb-[clamp(3rem,8vw,6rem)] px-[clamp(1.5rem,5vw,4rem)] overflow-hidden animate-slide-up">
      <div className="max-w-[1400px] w-full mx-auto relative z-10 flex items-center min-h-[70vh]">
        {/* Foreground Content: Text */}
        <div className="flex flex-col space-y-8 z-20 max-w-2xl relative bg-black/40 md:bg-transparent p-6 md:p-0 rounded-2xl md:rounded-none backdrop-blur-sm md:backdrop-blur-none border border-white/5 md:border-transparent">
          <h1 className="text-[clamp(2.5rem,6vw,5.5rem)] font-bold leading-[1.05] tracking-[-0.02em]">
            Learning <br />
            Conjunction Dynamics.
          </h1>

          <p className="text-base md:text-lg text-[#a0a0a0] max-w-md font-mono leading-relaxed tracking-wider">
            A Self-Supervised Approach to Satellite Collision Risk Assessment. Predicting future CDMs without collision labels.
          </p>


          <div className="flex flex-wrap gap-x-6 gap-y-2 font-mono text-[10px] tracking-widest text-[#606060] uppercase border-t border-white/5 pt-4">
            <span><span className="text-white">244,171</span> Parameters</span>
            <span><span className="text-white">185,511</span> CDMs</span>
            <span><span className="text-white">150</span> Epochs</span>
            <span><span className="text-white">2,003</span> Events</span>
          </div>
          <div className="pt-4 flex flex-row flex-nowrap gap-4 items-center">
            <a href="https://github.com/AhmedAlharbii/SDC2026_KAU_AE_TEAM" target="_blank" rel="noopener noreferrer" className="bg-white text-black px-6 md:px-8 py-3 md:py-4 font-mono text-xs tracking-[0.15em] font-bold hover:scale-105 transition-transform uppercase whitespace-nowrap text-center">
              [ View GitHub ]
            </a>
            <Link to="/paper" className="border border-white/20 text-white px-6 md:px-8 py-3 md:py-4 font-mono text-xs tracking-[0.15em] font-bold hover:bg-white/5 transition-colors uppercase whitespace-nowrap text-center">
              [ Read Paper ]
            </Link>
          </div>
        </div>
      </div>

      {/* Background/Corner Visual: Static Orbital Ring with one minimal rotation */}
      <div className="absolute right-[-20%] top-1/2 -translate-y-1/2 pointer-events-none z-0 opacity-50 md:opacity-80 flex justify-center items-center mix-blend-screen w-[800px] h-[800px] md:w-[1200px] md:h-[1200px]">
         <div className="relative flex justify-center items-center w-full h-full">
            <div className="relative overflow-hidden z-20 w-24 h-24 md:w-32 md:h-32 rounded-full flex items-center justify-center" style={{ background: '#000000', border: '1px solid rgba(255,255,255,0.1)', boxShadow: 'inset 0 0 40px rgba(255,255,255,0.05), 0 0 60px rgba(0,255,136,0.08)' }}>
               <div className="absolute inset-0 rounded-full" style={{ background: 'radial-gradient(circle at 35% 35%, rgba(255,255,255,0.15) 0%, transparent 60%)' }}></div>
            </div>
            
            {/* Highly optimized Static SVG with minimal inline animation */}
            <svg width="100%" height="100%" viewBox="0 0 1200 1200" className="absolute top-0 left-0 w-full h-full z-10 overflow-visible">
               {/* Static Rings */}
               <circle cx="600" cy="600" r="150" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="4 8" />
               <circle cx="600" cy="600" r="250" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
               <circle cx="600" cy="600" r="350" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
               <circle cx="600" cy="600" r="450" fill="none" stroke="rgba(255,255,255,0.02)" strokeWidth="1" />
               <circle cx="600" cy="600" r="550" fill="none" stroke="rgba(255,255,255,0.01)" strokeWidth="1" strokeDasharray="2 12" />
               
               {/* Static Nodes */}
               <circle cx="850" cy="600" r="3" fill="#fff" opacity="0.5" />
               <circle cx="250" cy="600" r="2" fill="#fff" opacity="0.3" />
               <circle cx="600" cy="150" r="4" fill="#ffbd2e" opacity="0.6" />
               <circle cx="600" cy="1050" r="2" fill="#fff" opacity="0.2" />
               
               {/* Single Animated Ring to keep it alive but performant */}
               <g style={{ animation: 'spin 60s linear infinite', transformOrigin: 'center' }}>
                  <circle cx="600" cy="600" r="300" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" strokeDasharray="100 800" />
                  <circle cx="900" cy="600" r="5" fill="#ff5a5a" />
                  <circle cx="300" cy="600" r="3" fill="#fff" />
                  <circle cx="600" cy="300" r="2" fill="#00ff88" />
               </g>
            </svg>
         </div>
      </div>
    </section>
  );
}
