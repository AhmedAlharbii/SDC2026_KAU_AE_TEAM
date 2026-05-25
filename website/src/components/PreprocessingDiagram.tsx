import { FileText, ArrowRight, Database, Settings, Layers, Box, Cpu } from 'lucide-react';
import { motion } from 'motion/react';

export default function PreprocessingDiagram() {
  const flow = [
    {
      icon: <FileText className="w-6 h-6 text-[#00ff88]" />,
      title: "Raw KVN Data",
      desc: "Chronological CDM parsing,\nUnit stripping & Object matching."
    },
    {
      icon: <Database className="w-6 h-6 text-[#00d4ff]" />,
      title: "CSV Dataset",
      desc: "Structured schema with 185k rows.\nNull handling & event grouping."
    },
    {
      icon: <Settings className="w-6 h-6 text-[#ff5a5a]" />,
      title: "Feature Engineering",
      desc: "Log10(Pc), Log1p Covariance,\nStandardScaler mapping."
    },
    {
      icon: <Layers className="w-6 h-6 text-[#ffbd2e]" />,
      title: "Sequence Generation",
      desc: "Sliding window extraction,\nVariable lengths (2 to 20)."
    },
    {
      icon: <Box className="w-6 h-6 text-[#b070ff]" />,
      title: "Padding & Masking",
      desc: "Left padded with -999.0,\nMasking layer skips nulls."
    }
  ];

  return (
    <div className="bg-[#050505] border border-white/10 p-6 rounded-xl my-8 relative overflow-hidden">
      <div className="absolute top-0 right-0 p-3 opacity-20 pointer-events-none">
        <Cpu className="w-32 h-32 text-white" />
      </div>
      <h4 className="text-white font-mono text-sm mb-8 tracking-widest uppercase">Data Pipeline Architecture</h4>
      
      <div className="flex flex-col md:flex-row items-start justify-between gap-8 md:gap-2 relative z-10">
        {flow.map((step, idx) => (
          <div key={idx} className="flex flex-col md:flex-row items-center flex-1 w-full relative">
            <div className="flex flex-col items-center text-center w-full px-2">
              <motion.div 
                initial={{ scale: 0.9, opacity: 0 }}
                whileInView={{ scale: 1, opacity: 1 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="w-16 h-16 rounded-full bg-[#111] border border-white/10 flex items-center justify-center shadow-[0_0_15px_rgba(255,255,255,0.05)] relative z-10 shrink-0"
              >
                {step.icon}
              </motion.div>
              
              <div className="mt-4 flex flex-col items-center flex-1">
                <div className="text-white text-xs font-bold font-mono uppercase mb-2 min-h-[2rem] flex items-center justify-center">{step.title}</div>
                <div className="text-[#888] text-[10px] uppercase font-mono tracking-wider leading-relaxed bg-black/40 p-2 rounded border border-white/5 w-full">
                  {step.desc.split('\n').map((line, i) => <div key={i}>{line}</div>)}
                </div>
              </div>
            </div>

            {idx < flow.length - 1 && (
              <div className="hidden md:flex items-center justify-center shrink-0 w-8 mx-1 opacity-50 relative top-[-1rem]">
                <ArrowRight className="text-white w-4 h-4" />
              </div>
            )}
            {idx < flow.length - 1 && (
              <div className="flex md:hidden h-8 w-px bg-gradient-to-b from-white/20 to-white/0 my-4"></div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
