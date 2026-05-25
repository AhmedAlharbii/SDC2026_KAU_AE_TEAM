import { ArrowRight, Database, Settings, Activity, ShieldCheck, LayoutDashboard, BarChart3, FileLineChart } from 'lucide-react';
import { motion } from 'motion/react';

export default function PipelineDiagram() {
  const steps = [
    { id: '1', icon: <Database className="w-5 h-5 text-[#00ff88]" />, title: "Parse KVN", active: true },
    { id: '2', icon: <Settings className="w-5 h-5 text-[#00d4ff]" />, title: "Prepare Sequences" },
    { id: '3', icon: <Activity className="w-5 h-5 text-[#ff5a5a]" />, title: "Train Model" },
    { id: '3B', icon: <ShieldCheck className="w-5 h-5 text-[#ffbd2e]" />, title: "Evaluate Proxy", sub: true },
    { id: '4', icon: <LayoutDashboard className="w-5 h-5 text-[#b070ff]" />, title: "Inference Dash" },
    { id: '5', icon: <BarChart3 className="w-5 h-5 text-[#ff8800]" />, title: "Visualize" },
    { id: '5B', icon: <FileLineChart className="w-5 h-5 text-[#00ff88]" />, title: "Detailed Reports", sub: true }
  ];

  return (
    <div className="bg-[#050505] border border-white/10 p-6 rounded-xl my-8 overflow-hidden">
      <h4 className="text-white font-mono text-sm mb-8 tracking-widest uppercase text-center md:text-left">The 7-Step Pipeline Ecosystem</h4>
      
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 relative z-10 w-full overflow-x-auto pb-4 hide-scrollbar">
        {steps.map((step, idx) => (
          <div key={step.id} className="flex flex-col items-center md:flex-row min-w-max relative">
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              whileInView={{ scale: 1, opacity: 1 }}
              transition={{ delay: idx * 0.1 }}
              viewport={{ once: true }}
              className={`flex flex-col items-center justify-center p-3 rounded-lg border ${step.sub ? 'border-dashed border-white/30 bg-[#111]' : 'border-white/10 bg-[#0a0a0a]'} min-w-[120px] shadow-[0_0_10px_rgba(255,255,255,0.02)]`}
            >
              <div className="mb-2">{step.icon}</div>
              <div className="text-white text-[10px] font-bold font-mono uppercase mb-0.5 whitespace-nowrap">{step.title}</div>
              <div className="text-[#666] text-[9px] font-mono">Step {step.id}</div>
            </motion.div>

            {idx < steps.length - 1 && (
              <div className="hidden md:flex items-center justify-center w-8 h-px mx-2 bg-white/20">
                <ArrowRight className="text-white/50 w-3 h-3 translate-x-3" />
              </div>
            )}
            {idx < steps.length - 1 && (
              <div className="flex md:hidden items-center justify-center h-6 w-px my-1 bg-white/20">
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
