import {
  Chart as ChartJS,
  registerables
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import { motion } from 'motion/react';
import { scatterData, highPriorityEvents, summaryStats, quadrantCounts } from '../data';

ChartJS.register(...registerables);

export default function Dashboard() {
  
  const scatterChartData = {
    datasets: [
      {
        label: 'ACT (Danger)',
        data: scatterData.filter(d => d.quadrant === 'ACT NOW').map(d => ({ x: d.threat, y: d.confidence, ...d })),
        backgroundColor: '#ff5a5a',
        pointHoverRadius: 8,
      },
      {
        label: 'WATCH (Warning)',
        data: scatterData.filter(d => d.quadrant === 'WATCH CLOSELY').map(d => ({ x: d.threat, y: d.confidence, ...d })),
        backgroundColor: '#ffbd2e',
        pointHoverRadius: 8,
      },
      {
        label: 'SAFE (Success)',
        data: scatterData.filter(d => d.quadrant === 'SAFELY IGNORE').map(d => ({ x: d.threat, y: d.confidence, ...d })),
        backgroundColor: '#00ff88',
        pointHoverRadius: 8,
      },
      {
        label: 'N/A',
        data: scatterData.filter(d => d.quadrant === 'NOT PRIORITY').map(d => ({ x: d.threat, y: d.confidence, ...d })),
        backgroundColor: '#808080',
        pointHoverRadius: 8,
      }
    ]
  };

  const scatterOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { 
        legend: { display: false }, 
        tooltip: { 
            backgroundColor: 'rgba(5, 5, 5, 0.95)',
            titleFont: { family: "'JetBrains Mono', monospace", size: 13, weight: 'bold' },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
            borderColor: 'rgba(255, 255, 255, 0.15)',
            borderWidth: 1,
            padding: 16,
            cornerRadius: 8,
            displayColors: true,
            boxPadding: 4,
            callbacks: {
                title: (context: any) => {
                    return `EVENT: ${context[0].raw.id}`;
                },
                label: (context: any) => {
                    const data = context.raw;
                    return [
                        `Objects: ${data.obj1} / ${data.obj2}`,
                        `Threat:  ${data.x.toFixed(2)}`,
                        `Conf:    ${(data.y * 100).toFixed(0)}%`
                    ];
                }
            }
        } 
    },
    scales: {
      x: { 
          title: { display: true, text: 'Threat Score â†’', color: 'rgba(255,255,255,0.4)', font: { family: 'JetBrains Mono', size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' },
          min: 0, max: 100
      },
      y: { 
          title: { display: true, text: 'Confidence â†’', color: 'rgba(255,255,255,0.4)', font: { family: 'JetBrains Mono', size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' },
          min: 0, max: 1.0
      }
    },
    animation: false as const
  };

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
    <section id="dashboard" className="pt-[clamp(4rem,10vw,8rem)] pb-[clamp(3rem,8vw,6rem)] px-[clamp(1.5rem,5vw,4rem)] border-t border-[#ffffff]/[0.08] overflow-hidden animate-slide-up">
      <motion.div 
        className="max-w-[1400px] mx-auto flex flex-col space-y-8"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={staggerContainer}
      >
        
        <motion.div variants={fadeInUp} className="flex justify-between items-center">
             <h2 className="text-[clamp(2.5rem,5vw,3.5rem)] font-bold leading-tight tracking-[-0.02em] text-white">
                 Live Feed.
             </h2>
        </motion.div>

        {/* Dashboard Frame Mockup */}
        <motion.div variants={fadeInUp} className="border border-[#ffffff]/[0.15] bg-[#000000] rounded-xl overflow-hidden shadow-[0_20px_50px_rgba(0,0,0,0.5)] relative">
            
            {/* Faux Browser Header */}
            <div className="bg-[#111111] border-b border-[#ffffff]/[0.1] px-4 py-3 flex items-center relative z-20">
                <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-[#ff5f56] border border-[#e0443e]"></div>
                    <div className="w-3 h-3 rounded-full bg-[#ffbd2e] border border-[#dea123]"></div>
                    <div className="w-3 h-3 rounded-full bg-[#27c93f] border border-[#1aab29]"></div>
                </div>
                <div className="absolute left-1/2 -translate-x-1/2 w-full max-w-[200px] md:max-w-xs">
                    <div className="bg-[#050505] border border-[#ffffff]/[0.08] text-[#808080] font-sans text-[11px] md:text-sm px-6 py-1 md:py-1.5 rounded-md text-center flex items-center justify-center gap-2 shadow-inner">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="opacity-50"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                        dashboard.debris-solver.io
                    </div>
                </div>
            </div>

            <div className="p-5 md:p-8 relative bg-[#050505]">

                {/* Gauges & Summary Top Row */}
            <motion.div variants={staggerContainer} className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 md:mb-8">
                <motion.div variants={fadeInUp} className="border border-[#ffffff]/[0.08] p-4 flex flex-col justify-between min-h-[100px] hover:bg-white/[0.02] transition-colors">
                    <span className="font-mono text-[10px] tracking-[0.15em] text-[#808080] uppercase">EVENTS</span>
                    <span className="font-mono text-2xl md:text-3xl text-white">{summaryStats.events.toLocaleString()}</span>
                </motion.div>
                <motion.div variants={fadeInUp} className="border border-[#ff5a5a]/30 p-4 flex flex-col justify-between min-h-[100px] relative overflow-hidden group hover:bg-[#ff5a5a]/5 transition-colors">
                    <div className="absolute top-0 right-0 w-8 h-8 bg-[#ff5a5a]/10 rounded-bl-full group-hover:scale-150 transition-transform"></div>
                    <span className="font-mono text-[10px] tracking-[0.15em] text-[#ff5a5a] uppercase relative z-10">ACT NOW</span>
                    <span className="font-mono text-2xl md:text-3xl text-[#ff5a5a] relative z-10">{quadrantCounts.act.count}</span>
                </motion.div>
                <motion.div variants={fadeInUp} className="border border-[#ffbd2e]/30 p-4 flex flex-col justify-between min-h-[100px] hover:bg-[#ffbd2e]/5 transition-colors">
                    <span className="font-mono text-[10px] tracking-[0.15em] text-[#ffbd2e] uppercase">WATCH</span>
                    <span className="font-mono text-2xl md:text-3xl text-[#ffbd2e]">{quadrantCounts.watch.count}</span>
                </motion.div>
                <motion.div variants={fadeInUp} className="border border-[#00ff88]/30 p-4 flex flex-col justify-between min-h-[100px] hover:bg-[#00ff88]/5 transition-colors">
                    <span className="font-mono text-[10px] tracking-[0.15em] text-[#00ff88] uppercase">SAFE</span>
                    <span className="font-mono text-2xl md:text-3xl text-[#00ff88]">{quadrantCounts.safe.count}</span>
                </motion.div>
            </motion.div>

            {/* Main Visualizer & Table Row */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-auto lg:h-[450px]">
                
                {/* Conjunction Map Chart */}
                <motion.div variants={fadeInUp} className="col-span-1 lg:col-span-8 border border-[#ffffff]/[0.08] p-4 flex flex-col relative min-h-[300px] lg:h-full group hover:border-[#ffffff]/[0.15] transition-colors">
                    <div className="font-mono text-[10px] tracking-[0.15em] text-[#808080] uppercase mb-4 flex items-center justify-between">
                       <span><span className="text-white font-bold">CONJUNCTION MAP</span> / THREAT VS CONFIDENCE</span>
                    </div>
                    <div className="flex-1 w-full h-full relative">
                        <Scatter data={scatterChartData} options={scatterOptions as any} />
                    </div>
                </motion.div>

                {/* High Priority Objects Table */}
                <motion.div variants={fadeInUp} className="col-span-1 lg:col-span-4 border border-[#ffffff]/[0.08] p-0 flex flex-col relative min-h-[300px] lg:h-full bg-[#050505]">
                    <div className="font-mono text-[10px] md:text-xs tracking-[0.15em] text-white font-bold uppercase border-b border-[#ffffff]/[0.08] bg-white/[0.03] p-4">
                        HIGH PRIORITY INTERSECTIONS
                    </div>
                    <div className="flex-1 overflow-auto custom-scrollbar p-1">
                        <table className="w-full text-left font-mono text-xs md:text-sm">
                          <thead className="sticky top-0 bg-[#050505] z-10">
                            <tr className="text-[#a0a0a0]">
                              <th className="py-3 px-3 font-medium tracking-wider uppercase border-b border-[#ffffff]/[0.08] bg-[#050505]">Event Pair</th>
                              <th className="py-3 px-3 font-medium tracking-wider uppercase text-right border-b border-[#ffffff]/[0.08] bg-[#050505]">Threat</th>
                              <th className="py-3 px-3 font-medium tracking-wider uppercase text-right border-b border-[#ffffff]/[0.08] bg-[#050505]">Conf</th>
                            </tr>
                          </thead>
                          <tbody>
                              {highPriorityEvents.slice(0, 10).map((evt, i) => (
                                  <tr key={i} className="hover:bg-white/[0.04] transition-all duration-200 cursor-crosshair group">
                                      <td className="py-3 px-3 text-white truncate max-w-[120px] md:max-w-none border-b border-[#ffffff]/[0.04] group-hover:text-[#00ff88]">
                                          {evt.obj1} <span className="opacity-40 text-[#a0a0a0] group-hover:text-white transition-colors">/</span> {evt.obj2}
                                      </td>
                                      <td className="py-3 px-3 text-right border-b border-[#ffffff]/[0.04] text-[#ff5a5a] font-bold">{evt.ts.toFixed(1)}</td>
                                      <td className="py-3 px-3 text-right text-gray-300 border-b border-[#ffffff]/[0.04]">{(evt.cnf * 100).toFixed(0)}%</td>
                                  </tr>
                              ))}
                          </tbody>
                        </table>
                    </div>
                </motion.div>
            </div>
            </div>
        </motion.div>
      </motion.div>
    </section>
  );
}

