import { motion } from 'motion/react';
import { quadrantCounts } from '../data';

export default function Solution() {
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
        className="max-w-[1400px] w-full mx-auto flex flex-col space-y-16"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={staggerContainer}
      >
        
        {/* Concept Text */}
        <motion.div variants={fadeInUp} className="text-center space-y-6">
            <h2 className="font-mono text-[clamp(1.5rem,4vw,2.5rem)] font-bold text-white mb-6">
                Breaking the Threshold Bias
            </h2>
            <div className="font-mono text-sm md:text-base text-[#a0a0a0] max-w-4xl mx-auto leading-relaxed border-b border-[#ffffff]/[0.08] pb-12 space-y-4">
                <p>The foundational insight of DebriSolver is simple but powerful: a sequence of CDMs for a single conjunction event is not a collection of independent snapshots.</p>
                <p><strong className="text-white">It is a story with a trajectory</strong>, and that trajectory is rich with information. Is the Pc curve rising? Is the covariance collapsing? Is it simply tracking noise that will resolve itself?</p>
                <p>By training a BiGRU on thousands of CDM sequences in a self-supervised loop, we learn to predict the next CDM without ever being given explicit collision labels.</p>
            </div>
            
            <div className="font-mono text-white text-xl md:text-3xl py-10 flex flex-wrap justify-center items-center gap-3 md:gap-5 w-full">
                 <span className="text-white/20 hidden md:inline tracking-widest font-bold">──────</span>
                 <span className="text-white/50 tracking-widest">●──●──●──●──</span>
                 <span className="tracking-widest">●──●──</span>
                 <span className="text-[#00ff88] bg-[#00ff88]/10 px-5 py-2 border border-[#00ff88]/50 shadow-[0_0_25px_rgba(0,255,136,0.15)] font-bold rounded-sm ml-2">
                    [ ? ]
                 </span>
                 <span className="text-white/20 hidden md:inline ml-3 font-bold">───→</span>
            </div>
            <div className="font-mono text-xs md:text-sm text-[#808080] tracking-widest flex justify-center gap-12 md:gap-32 uppercase relative pb-4">
                 <span>CDM₁ ... CDMₙ</span>
                 <span className="text-[#00ff88] bg-[#000000] px-3 py-1 border border-[#00ff88]/30 rounded shadow-[0_0_10px_rgba(0,255,136,0.1)]">Predicted CDMₙ₊₁</span>
            </div>
        </motion.div>

        {/* 4 Quadrant Grid */}
        <motion.div variants={fadeInUp} className="max-w-4xl mx-auto w-full pt-4 space-y-4 text-center px-4">
            <h3 className="font-mono text-sm md:text-base text-white font-bold uppercase tracking-widest mb-4">Confidence Calibration</h3>
            <p className="font-mono text-xs md:text-sm text-[#a0a0a0]">
                MC Dropout Uncertainty (40%) + Data Quantity (35%) + Covariance Quality (25%)
            </p>
            <p className="font-mono text-xs md:text-sm text-[#00ff88] pt-2 pb-6">
                Operators reduced urgent attention burden by 84% by focusing only on ACT NOW events.
            </p>
        </motion.div>
        
        <motion.div variants={staggerContainer} className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto w-full pt-2">
            
            {/* ACT NOW */}
            <motion.div variants={fadeInUp} className="quadrant-card quadrant--danger hover:scale-[1.02] transition-transform">
                <div className="flex justify-between items-center w-full">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-[#ff5a5a] shadow-[0_0_8px_#ff5a5a]"></span>
                        <span className="q-label text-xs md:text-sm text-white/50 tracking-widest uppercase">ACT NOW</span>
                    </div>
                    <span className="q-count text-white font-bold text-lg">{quadrantCounts.act.count}</span>
                </div>
                <div className="font-mono text-xs md:text-sm text-[#a0a0a0] flex flex-col gap-1 mt-6">
                    <span>↑ High Threat Score</span>
                    <span>↑ High Confidence</span>
                </div>
                <div className="text-xs md:text-sm border-t border-[#ffffff]/[0.08] pt-5 mt-5 text-[#808080]">
                    → Immediate maneuver evaluation
                </div>
            </motion.div>

            {/* WATCH CLOSELY */}
            <motion.div variants={fadeInUp} className="quadrant-card quadrant--warning hover:scale-[1.02] transition-transform">
                <div className="flex justify-between items-center w-full">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-[#ffbd2e] shadow-[0_0_8px_#ffbd2e]"></span>
                        <span className="q-label text-xs md:text-sm text-white/50 tracking-widest uppercase">WATCH CLOSELY</span>
                    </div>
                    <span className="q-count text-white font-bold text-lg">{quadrantCounts.watch.count}</span>
                </div>
                <div className="font-mono text-xs md:text-sm text-[#a0a0a0] flex flex-col gap-1 mt-6">
                    <span>↑ High Threat Score</span>
                    <span>↓ Low Confidence</span>
                </div>
                <div className="text-xs md:text-sm border-t border-[#ffffff]/[0.08] pt-5 mt-5 text-[#808080]">
                    → Request tracking update
                </div>
            </motion.div>

            {/* SAFELY IGNORE */}
            <motion.div variants={fadeInUp} className="quadrant-card quadrant--success hover:scale-[1.02] transition-transform">
                <div className="flex justify-between items-center w-full">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-[#00ff88] shadow-[0_0_8px_#00ff88]"></span>
                        <span className="q-label text-xs md:text-sm text-white/50 tracking-widest uppercase">SAFELY IGNORE</span>
                    </div>
                    <span className="q-count text-white font-bold text-lg">{quadrantCounts.safe.count}</span>
                </div>
                <div className="font-mono text-xs md:text-sm text-[#a0a0a0] flex flex-col gap-1 mt-6">
                    <span>↓ Low Threat Score</span>
                    <span>↑ High Confidence</span>
                </div>
                <div className="text-xs md:text-sm border-t border-[#ffffff]/[0.08] pt-5 mt-5 text-[#808080]">
                    → Deprioritize
                </div>
            </motion.div>

            {/* NOT PRIORITY */}
            <motion.div variants={fadeInUp} className="quadrant-card quadrant--neutral hover:scale-[1.02] transition-transform">
                <div className="flex justify-between items-center w-full">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-[#808080]"></span>
                        <span className="q-label text-xs md:text-sm text-white/50 tracking-widest uppercase">NOT PRIORITY</span>
                    </div>
                    <span className="q-count text-white font-bold text-lg">{quadrantCounts.na.count}</span>
                </div>
                <div className="font-mono text-xs md:text-sm text-[#a0a0a0] flex flex-col gap-1 mt-6">
                    <span>↓ Low Threat Score</span>
                    <span>↓ Low Confidence</span>
                </div>
                <div className="text-xs md:text-sm border-t border-[#ffffff]/[0.08] pt-5 mt-5 text-[#808080]">
                    → Routine monitor
                </div>
            </motion.div>
        </motion.div>

      </motion.div>
    </section>
  );
}
