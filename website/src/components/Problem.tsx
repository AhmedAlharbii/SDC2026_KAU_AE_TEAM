import { motion } from 'motion/react';
import { eventsData, summaryStats } from '../data';

export default function Problem() {

  const fadeInUp = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" as any } }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  return (
    <section className="pt-[clamp(4rem,10vw,8rem)] pb-[clamp(3rem,8vw,6rem)] px-[clamp(1.5rem,5vw,4rem)] border-t border-[#ffffff]/[0.08] overflow-hidden animate-slide-up">
      <motion.div 
        className="max-w-[1400px] w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-16"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={staggerContainer}
      >
        
        {/* Left Side: Text */}
        <motion.div variants={fadeInUp} className="lg:col-span-5 flex flex-col space-y-10">
          <div>
            <h2 className="text-[clamp(2.5rem,5vw,3.75rem)] font-bold leading-tight tracking-[-0.02em] text-white">
              The Alert Overload Problem.
            </h2>
          </div>
          
          <div className="space-y-6 text-[#a0a0a0] font-mono text-sm md:text-base leading-relaxed max-w-xl tracking-wide pr-0 md:pr-4">
             <p>Every 90 minutes, the ISS completes one orbit through 27,000 tracked pieces of junk - and millions more too small to see.</p>
             <p>A constellation operator may receive 50k-200k conjunction screening alerts annually. Traditional threshold-based systems (like a standard Pc {'>'} 1×10⁻⁴) ignore track uncertainty and trajectory evolution, resulting in massive <span className="text-white bg-white/10 px-2 py-0.5 rounded">Alert Fatigue</span>.</p>
             <p>The Iridium-Cosmos 2009 collision proved the danger: creating 2,000+ new fragments. The cost of missing a real threat is devastating. The cost of acting unnecessarily is wasted fuel. We need to triage efficiently.</p>
          </div>
        </motion.div>

        {/* Right Side: Visual & Data Table */}
        <div className="lg:col-span-7 lg:pt-12 flex flex-col gap-6">

          {/* Animated GEO Ring Visualization */}
          <motion.div variants={fadeInUp} className="border border-[#ffffff]/[0.08] bg-[#050505] relative h-[220px] md:h-[300px] overflow-hidden flex justify-center items-center group">
            
            <div className="geo-ring-container relative flex justify-center items-center scale-75 md:scale-100 group-hover:scale-105 transition-transform duration-700 w-full h-full">
               <div className="planet-core relative overflow-hidden z-20 w-12 h-12 rounded-full flex items-center justify-center bg-black" style={{ border: '1px solid rgba(255,255,255,0.1)', boxShadow: 'inset 0 0 15px rgba(255,255,255,0.05), 0 0 30px rgba(0,255,136,0.05)' }}>
                  <div className="absolute inset-0 rounded-full" style={{ background: 'radial-gradient(circle at 35% 35%, rgba(255,255,255,0.05) 0%, transparent 60%)' }}></div>
               </div>
               
               {/* Highly optimized SVG with mostly static parts and single rotation */}
               <svg width="400" height="400" viewBox="0 0 400 400" className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10 overflow-visible opacity-90">
                  {/* Static Orbits */}
                  {Array.from({ length: 6 }).map((_, i) => (
                     <circle key={i} cx="200" cy="200" r={40 + i * 25} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" strokeDasharray={i % 2 === 0 ? "4 4" : "none"} />
                  ))}
                  
                  {/* Static Debris Nodes */}
                  <circle cx="200" cy="90" r="2" fill="#fff" opacity="0.4" />
                  <circle cx="280" cy="200" r="1.5" fill="#fff" opacity="0.6" />
                  <circle cx="120" cy="250" r="2.5" fill="#ffbd2e" opacity="0.8" />
                  <circle cx="320" cy="100" r="2" fill="#ff5a5a" />
                  
                  {/* Single animated element */}
                  <g style={{ animation: `spin 30s linear infinite`, transformOrigin: 'center' }}>
                     <circle cx="200" cy="200" r="140" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="60 300" />
                     <circle cx="340" cy="200" r="3" fill="#ff5a5a" />
                     <circle cx="60" cy="200" r="2" fill="#fff" opacity="0.5" />
                  </g>
               </svg>
            </div>

            <div className="absolute bottom-4 left-4 font-mono text-[9px] tracking-widest text-[#808080] uppercase z-10 bg-[#000000]/80 px-2 py-1">
               Geosynchronous ring of debris
            </div>
          </motion.div>

          {/* Core Problem Stats */}
          <motion.div variants={staggerContainer} className="grid grid-cols-2 md:grid-cols-4 gap-4">
             <motion.div variants={fadeInUp} className="border border-[#ffffff]/[0.08] p-4 bg-[#050505] hover:bg-white/[0.02] transition-colors">
                 <div className="text-xl md:text-2xl text-white font-mono">27,000</div>
                 <div className="text-[10px] text-[#a0a0a0] font-mono mt-1 uppercase tracking-widest leading-tight">Tracked objects</div>
             </motion.div>
             <motion.div variants={fadeInUp} className="border border-[#ffffff]/[0.08] p-4 bg-[#050505] hover:bg-white/[0.02] transition-colors">
                 <div className="text-xl md:text-2xl text-white font-mono">1M+</div>
                 <div className="text-[10px] text-[#a0a0a0] font-mono mt-1 uppercase tracking-widest leading-tight">Unseen threats</div>
             </motion.div>
             <motion.div variants={fadeInUp} className="border border-[#ffffff]/[0.08] p-4 bg-[#050505] hover:bg-white/[0.02] transition-colors">
                 <div className="text-xl md:text-2xl text-white font-mono">50k+</div>
                 <div className="text-[10px] text-[#a0a0a0] font-mono mt-1 uppercase tracking-widest leading-tight">Alerts/year</div>
             </motion.div>
             <motion.div variants={fadeInUp} className="border border-[#ff5a5a]/30 p-4 bg-[#ff5a5a]/5 hover:bg-[#ff5a5a]/10 transition-colors">
                 <div className="text-xl md:text-2xl text-[#ff5a5a] font-mono">1:1M</div>
                 <div className="text-[10px] text-[#ff5a5a] font-mono mt-1 uppercase tracking-widest leading-tight">Collision Ratio</div>
             </motion.div>
          </motion.div>

          {/* Structural Limitations */}
          <motion.div variants={fadeInUp} className="border border-[#ffffff]/[0.08] bg-[#050505] p-6 lg:p-10 flex flex-col relative overflow-hidden text-sm md:text-base text-[#a0a0a0] font-mono shadow-xl rounded-sm">
             <h3 className="text-white text-sm md:text-base font-bold mb-6 tracking-widest uppercase pb-4 border-b border-[#ffffff]/[0.08]">
                 Why Existing Systems Fail
             </h3>
             <ul className="space-y-6">
               <li className="flex gap-5 items-start bg-black/40 p-4 border border-white/5 rounded">
                  <span className="text-white bg-white/5 px-2 py-1 rounded text-xs">01.</span>
                  <div className="tracking-wide"><strong className="text-white font-normal block mb-1">Threshold-based.</strong> Flags blindly when Pc {'>'} 1×10⁻⁴, ignoring trajectory. Is risk rising or falling?</div>
               </li>
               <li className="flex gap-5 items-start bg-black/40 p-4 border border-white/5 rounded">
                  <span className="text-white bg-white/5 px-2 py-1 rounded text-xs">02.</span>
                  <div className="tracking-wide"><strong className="text-white font-normal block mb-1">Zero confidence estimation.</strong> A 100km covariance (poor track) looks identical to a 10m covariance (perfect track).</div>
               </li>
               <li className="flex gap-5 items-start bg-black/40 p-4 border border-white/5 rounded">
                  <span className="text-white bg-white/5 px-2 py-1 rounded text-xs">03.</span>
                  <div className="tracking-wide"><strong className="text-white font-normal block mb-1">Treat CDMs in isolation.</strong> The rich story contained inside the evolution of 10 CDMs over an 8-day timeline is ignored.</div>
               </li>
             </ul>
          </motion.div>
        </div>

      </motion.div>
    </section>
  );
}
