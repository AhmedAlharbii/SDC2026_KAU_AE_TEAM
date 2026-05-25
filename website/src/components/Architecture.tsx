import { motion } from 'motion/react';
import GRUCellDiagram from './GRUCellDiagram';

export default function Architecture() {
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
    <section id="architecture" className="pt-[clamp(4rem,10vw,8rem)] pb-[clamp(3rem,8vw,6rem)] px-[clamp(1.5rem,5vw,4rem)] border-t border-[#ffffff]/[0.08] overflow-hidden animate-slide-up">
      <motion.div 
        className="max-w-[1400px] w-full mx-auto space-y-24"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={staggerContainer}
      >
        
        {/* Full Data Pipeline Diagram */}
        <div className="flex flex-col space-y-8">
          <motion.div variants={fadeInUp} className="flex flex-col md:flex-row justify-between items-start md:items-end mb-4">
            <div>
              <h2 className="text-[clamp(2rem,4vw,3rem)] font-bold leading-tight tracking-[-0.02em] text-white">
                The Pipeline.
              </h2>
              <p className="text-[#a0a0a0] font-mono text-sm md:text-base leading-relaxed max-w-md tracking-wide mt-4">
                End-to-end data flow from raw space tracking observations to prioritized dashboard outputs.
              </p>
            </div>
          </motion.div>

          <motion.div variants={fadeInUp} className="w-full border border-[#ffffff]/[0.08] bg-[#050505] p-6 lg:p-12 relative overflow-hidden group hover:border-[#ffffff]/[0.15] transition-colors">

            {/* Mobile/Tablet (Stack) View */}
            <div className="flex flex-col lg:hidden font-mono text-[10px] text-center relative z-10 space-y-8">
              <div className="absolute top-8 bottom-8 left-1/2 -translate-x-1/2 border-l-[2px] border-dashed border-white/20 -z-10"></div>
              
              <div className="flex flex-col items-center group/step hover:scale-105 transition-transform">
                <div className="w-16 h-16 rounded-full border border-white/20 bg-[#050505] flex items-center justify-center mb-4 z-10">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-white group-hover/step:text-[#00ff88] transition-colors">
                    <path d="M12 2v20M2 12h20M12 2a10 10 0 0 1 10 10M12 2a10 10 0 0 0-10 10M12 22a10 10 0 0 1-10-10M12 22a10 10 0 0 0 10-10" />
                  </svg>
                </div>
                <div className="text-white font-bold uppercase tracking-widest mb-1 bg-[#050505] px-2">1. Space Data</div>
                <div className="text-[#808080] bg-[#050505] px-2 mb-2">Sensors track objects<br/>CDMs generated</div>
                <div className="text-white/40">▼</div>
              </div>

              <div className="flex flex-col items-center group/step hover:scale-105 transition-transform">
                <div className="w-16 h-16 rounded border border-white/20 bg-[#050505] flex flex-col justify-center gap-1.5 mb-4 z-10 p-3">
                  <div className="h-1 bg-white/20 w-3/4 group-hover/step:bg-[#00ff88]/50 transition-colors"></div>
                  <div className="h-1 bg-[#00ff88]/50 w-full group-hover/step:bg-[#00ff88] transition-colors"></div>
                  <div className="h-1 bg-white/20 w-5/6 group-hover/step:bg-[#00ff88]/50 transition-colors"></div>
                </div>
                <div className="text-white font-bold uppercase tracking-widest mb-1 bg-[#050505] px-2">2. KVN Parser</div>
                <div className="text-[#808080] bg-[#050505] px-2 mb-2">Regex parsing<br/>Derived features</div>
                <div className="text-white/40">▼</div>
              </div>

              <div className="flex flex-col items-center group/step hover:scale-105 transition-transform">
                <div className="w-16 h-16 border border-white/20 bg-[#050505] flex items-center justify-center mb-4 z-10">
                  <div className="grid grid-cols-3 grid-rows-3 gap-[2px] w-8 h-8 group-hover/step:rotate-90 transition-transform duration-700">
                     {[...Array(9)].map((_, i) => <div key={i} className={`bg-white/${i<4 ? '20': (i===4?'60':'10')}`}></div>)}
                  </div>
                </div>
                <div className="text-white font-bold uppercase tracking-widest mb-1 bg-[#050505] px-2">3. Seq Tensor</div>
                <div className="text-[#808080] bg-[#050505] px-2 mb-2">Max 20 timesteps<br/>Sentinel padded</div>
                <div className="text-white/40">▼</div>
              </div>

              <div className="flex flex-col items-center group/step hover:scale-105 transition-transform">
                <div className="w-16 h-16 rounded-lg border border-white/40 bg-[#000000] flex items-center justify-center mb-4 z-10 shadow-[0_0_15px_rgba(255,255,255,0.1)] group-hover/step:shadow-[0_0_20px_rgba(0,255,136,0.2)] transition-shadow">
                   <span className="text-white text-[10px] font-bold">Engine</span>
                </div>
                <div className="text-white font-bold uppercase tracking-widest mb-1 bg-[#050505] px-2">4. AI Model</div>
                <div className="text-[#808080] bg-[#050505] px-2 mb-2">Self-supervised<br/>MC Dropout</div>
                <div className="text-white/40">▼</div>
              </div>

              <div className="flex flex-col items-center group/step hover:scale-105 transition-transform">
                <div className="w-16 h-16 rounded-full border border-[#ff5a5a]/50 bg-[#110505] flex items-center justify-center mb-4 z-10">
                  <div className="w-6 h-6 rounded-full border border-[#ffbd2e] bg-[#ffbd2e]/20 group-hover/step:bg-[#ff5a5a]/50 transition-colors"></div>
                </div>
                <div className="text-white font-bold uppercase tracking-widest mb-1 bg-[#050505] px-2">5. Output</div>
                <div className="text-[#808080] bg-[#050505] px-2">Quad Dashboard<br/>Threat & Conf</div>
              </div>
            </div>

            {/* Desktop (Linear) View */}
            <div className="hidden lg:flex flex-row flex-nowrap justify-between items-center font-mono xl:px-8 py-12 w-full max-w-6xl mx-auto border border-white/5 bg-[#050505]/50 rounded-xl relative shadow-[0_0_50px_rgba(0,0,0,0.5)]">
                
                {/* Step 1 */}
                <div className="flex flex-col items-center group/step hover:-translate-y-2 transition-transform cursor-default px-4 w-[160px] relative z-10 shrink-0">
                  <div className="w-16 h-16 rounded-full border border-white/20 bg-[#050505] flex items-center justify-center mb-4 z-10 group-hover/step:border-[#00ff88]/50 shadow-[0_0_15px_rgba(0,0,0,0.5)] group-hover/step:shadow-[0_0_20px_rgba(0,255,136,0.15)] transition-all">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-white group-hover/step:text-[#00ff88] transition-colors">
                      <path d="M12 2v20M2 12h20M12 2a10 10 0 0 1 10 10M12 2a10 10 0 0 0-10 10M12 22a10 10 0 0 1-10-10M12 22a10 10 0 0 0 10-10" />
                    </svg>
                  </div>
                  <div className="text-white font-bold uppercase tracking-widest mb-2 text-xs text-center">1. Space Data</div>
                  <div className="text-[#a0a0a0] text-xs text-center">Sensors track<br/>CDMs formed</div>
                </div>

                {/* Arrow */}
                <div className="text-white/20 shrink-0 px-2 lg:px-4">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14m-7-7 7 7-7 7"/></svg>
                </div>

                {/* Step 2 */}
                <div className="flex flex-col items-center group/step hover:-translate-y-2 transition-transform cursor-default px-4 w-[160px] relative z-10 shrink-0">
                  <div className="w-16 h-16 rounded border border-white/20 bg-[#050505] flex flex-col justify-center gap-1.5 mb-4 z-10 p-3 shadow-[0_0_15px_rgba(0,0,0,0.5)] group-hover/step:border-[#00ff88]/50 transition-all">
                    <div className="h-1 bg-white/20 w-3/4 group-hover/step:bg-[#00ff88]/50 transition-colors"></div>
                    <div className="h-1 bg-[#00ff88]/50 w-full group-hover/step:bg-[#00ff88] transition-colors"></div>
                    <div className="h-1 bg-white/20 w-5/6 group-hover/step:bg-[#00ff88]/50 transition-colors"></div>
                  </div>
                  <div className="text-white font-bold uppercase tracking-widest mb-2 text-xs text-center">2. KVN Parser</div>
                  <div className="text-[#a0a0a0] text-xs text-center">Regex parsing<br/>Features built</div>
                </div>

                {/* Arrow */}
                <div className="text-white/20 shrink-0 px-2 lg:px-4">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14m-7-7 7 7-7 7"/></svg>
                </div>

                {/* Step 3 */}
                <div className="flex flex-col items-center group/step hover:-translate-y-2 transition-transform cursor-default px-4 w-[160px] relative z-10 shrink-0">
                  <div className="w-16 h-16 border border-white/20 bg-[#050505] flex items-center justify-center mb-4 z-10 shadow-[0_0_15px_rgba(0,0,0,0.5)] group-hover/step:border-[#00ff88]/50 transition-all">
                    <div className="grid grid-cols-3 grid-rows-3 gap-[2px] w-8 h-8 group-hover/step:rotate-90 transition-transform duration-700">
                       {[...Array(9)].map((_, i) => <div key={i} className={`bg-white/${i<4 ? '20': (i===4?'60':'10')}`}></div>)}
                    </div>
                  </div>
                  <div className="text-white font-bold uppercase tracking-widest mb-2 text-xs text-center">3. Seq Tensor</div>
                  <div className="text-[#a0a0a0] text-xs text-center">20 timesteps<br/>Sentinel padded</div>
                </div>

                {/* Arrow */}
                <div className="text-white/20 shrink-0 px-2 lg:px-4">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14m-7-7 7 7-7 7"/></svg>
                </div>

                {/* Step 4 */}
                <div className="flex flex-col items-center group/step hover:-translate-y-2 transition-transform cursor-default px-4 w-[160px] relative z-10 shrink-0">
                  <div className="w-16 h-16 rounded-lg border border-white/40 bg-[#000000] flex items-center justify-center mb-4 z-10 shadow-[0_0_15px_rgba(0,255,136,0.1)] group-hover/step:shadow-[0_0_25px_rgba(0,255,136,0.3)] transition-all">
                     <span className="text-white text-xs font-bold">Engine</span>
                  </div>
                  <div className="text-white font-bold uppercase tracking-widest mb-2 text-xs text-center">4. AI Model</div>
                  <div className="text-[#a0a0a0] text-xs text-center">Self-supervised<br/>MC Dropout</div>
                </div>

                {/* Arrow */}
                <div className="text-white/20 shrink-0 px-2 lg:px-4">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14m-7-7 7 7-7 7"/></svg>
                </div>

                {/* Step 5 */}
                <div className="flex flex-col items-center group/step hover:-translate-y-2 transition-transform cursor-default px-4 w-[160px] relative z-10 shrink-0">
                  <div className="w-16 h-16 rounded-full border border-[#ff5a5a]/50 bg-[#110505] flex items-center justify-center mb-4 z-10 shadow-[0_0_15px_rgba(255,90,90,0.1)] group-hover/step:shadow-[0_0_20px_rgba(255,189,46,0.2)] transition-all">
                    <div className="w-6 h-6 rounded-full border border-[#ffbd2e] bg-[#ffbd2e]/20 group-hover/step:bg-[#ff5a5a]/50 transition-colors"></div>
                  </div>
                  <div className="text-white font-bold uppercase tracking-widest mb-2 text-xs text-center">5. Output</div>
                  <div className="text-[#a0a0a0] text-xs text-center">Quad Dashboard<br/>Threat & Conf</div>
                </div>
            </div>
          </motion.div>
        </div>

        {/* BiGRU Meta-Architecture diagram */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 border-t border-[#ffffff]/[0.08] pt-24 items-center">
          <motion.div variants={fadeInUp} className="lg:col-span-4 flex flex-col space-y-8 pr-4">
            <h2 className="text-[clamp(2.5rem,5vw,3.75rem)] font-bold leading-tight tracking-[-0.02em] text-white">
              The Architecture.
            </h2>
            <div className="space-y-6 text-[#a0a0a0] font-mono text-sm md:text-base leading-relaxed tracking-wide">
              <p>
                The core of DebriSolver is a Bidirectional Gated Recurrent Unit (BiGRU) composed of 244,171 parameters.
              </p>
              <ul className="space-y-6">
                <li className="flex gap-4 items-start">
                  <span className="text-white bg-white/5 rounded px-2 py-1 shrink-0 text-xs mt-0.5">01.</span>
                  <div><strong className="text-white font-normal block mb-1">Masking Layer</strong> Zeroes out gradient flow for padded sequence slots (-999.0 sentinels).</div>
                </li>
                <li className="flex gap-4 items-start">
                  <span className="text-white bg-white/5 rounded px-2 py-1 shrink-0 text-xs mt-0.5">02.</span>
                  <div><strong className="text-white font-normal block mb-1">Bidirectional GRU</strong> Learns temporal causal dynamics inside variable length event trajectories.</div>
                </li>
                <li className="flex gap-4 items-start">
                  <span className="text-white bg-white/5 rounded px-2 py-1 shrink-0 text-xs mt-0.5">03.</span>
                  <div><strong className="text-white font-normal block mb-1">LayerNorm & Dropout</strong> Enables MC Dropout at inference time to act as a Bayesian approximation of model epistemic uncertainty.</div>
                </li>
              </ul>
            </div>
          </motion.div>
          
          <motion.div variants={fadeInUp} className="lg:col-span-8 flex flex-col items-center w-full">
             <div className="w-full flex justify-center bg-[#050505] border border-[#ffffff]/[0.08] p-8 md:p-16 relative overflow-hidden group hover:border-[#ffffff]/[0.15] transition-colors rounded-xl mx-auto shadow-[0_15px_40px_rgba(0,0,0,0.4)]">
                
                <div className="group-hover:scale-[1.02] transition-transform duration-700 ease-out w-full max-w-2xl mx-auto">
                    <GRUCellDiagram />
                </div>
                
                <div className="absolute bottom-6 left-6 font-mono text-[10px] md:text-xs tracking-widest text-[#808080] lg:text-white/40 uppercase">
                   Gated Recurrent Unit Inner Mechanism
                </div>
             </div>
             
             {/* Simple vertical stack representing the full Network flow */}
             <div className="w-full mt-12 font-mono text-[9px] md:text-[10px] text-center grid grid-cols-5 gap-1 md:gap-2 opacity-80">
                <div className="border border-white/10 p-1 md:p-2 py-4 bg-[#050505] text-[#808080] hover:bg-white/[0.05] transition-colors cursor-crosshair">CDM<br/>N-20</div>
                <div className="border border-white/10 p-1 md:p-2 py-4 bg-[#050505] text-[#808080] hover:bg-white/[0.05] transition-colors cursor-crosshair">CDM<br/>N-19</div>
                <div className="p-1 md:p-2 py-4 flex items-center justify-center text-white/50 tracking-[0.2em]">...</div>
                <div className="border border-white/20 p-1 md:p-2 py-4 bg-[#ffffff]/5 text-white shadow-[0_0_10px_rgba(255,255,255,0.05)] cursor-crosshair">CDM<br/>N</div>
                <div className="border border-[#ff5a5a] p-1 md:p-2 py-4 bg-[#ff5a5a]/10 text-[#ff5a5a] relative cursor-crosshair hover:bg-[#ff5a5a]/20 transition-colors">PREDICT<br/>N+1<span className="hidden md:block absolute top-1/2 left-0 -translate-x-full h-[1px] w-4 bg-[#ff5a5a]"></span></div>
             </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
}
