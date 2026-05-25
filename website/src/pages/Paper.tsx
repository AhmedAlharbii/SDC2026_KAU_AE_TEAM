import { useEffect, useState, useMemo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import "katex/dist/katex.min.css";
import rehypeRaw from "rehype-raw";
import { ArrowLeft, Download, FileText, ArrowUp, ChevronDown, ChevronRight } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import GithubSlugger from 'github-slugger';
import { paperMarkdown } from "../data/paperContent";
import PreprocessingDiagram from "../components/PreprocessingDiagram";
import PipelineDiagram from "../components/PipelineDiagram";
import UncertaintyPlot from "../components/UncertaintyPlot";
import GRUCellDiagram from "../components/GRUCellDiagram";
import TeamCards from "../components/TeamCards";
import { 
  ThreatDistributionChart,
  ConfidenceDistributionChart,
  ThreatVsPcChart,
  ConfidenceVsCdmsChart,
  MaecurveChart,
  PcPredictionErrorChart,
  TrainingLossChart
} from "../components/Section13Charts";
import { scatterData } from "../data";
import { Line, Scatter } from "react-chartjs-2";
import { Chart as ChartJS, registerables } from "chart.js";
import { useScrollSpy } from "../hooks/useScrollSpy";

ChartJS.register(...registerables);


const quadrantLinesPlugin = {
  id: 'quadrantLinesPaper',
  afterDraw(chart: any) {
    const ctx = chart.ctx;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    if (!xScale || !yScale) return;
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.lineWidth = 1;
    const x50 = xScale.getPixelForValue(50);
    ctx.beginPath(); ctx.moveTo(x50, chart.chartArea.top); ctx.lineTo(x50, chart.chartArea.bottom); ctx.stroke();
    const y05 = yScale.getPixelForValue(0.5);
    ctx.beginPath(); ctx.moveTo(chart.chartArea.left, y05); ctx.lineTo(chart.chartArea.right, y05); ctx.stroke();
    ctx.restore();
  }
};

const QuadrantChart = () => {
  const data = {
    datasets: [
      { label: "ACT NOW",      data: scatterData.filter(d => d.quadrant === "ACT NOW").map(d => ({ x: d.threat, y: d.confidence })),      backgroundColor: "#ff5a5a", pointRadius: 4, pointHoverRadius: 7 },
      { label: "WATCH CLOSELY",data: scatterData.filter(d => d.quadrant === "WATCH CLOSELY").map(d => ({ x: d.threat, y: d.confidence })), backgroundColor: "#ffbd2e", pointRadius: 3, pointHoverRadius: 7 },
      { label: "SAFELY IGNORE",data: scatterData.filter(d => d.quadrant === "SAFELY IGNORE").map(d => ({ x: d.threat, y: d.confidence })), backgroundColor: "#00ff88", pointRadius: 3, pointHoverRadius: 7 },
      { label: "NOT PRIORITY", data: scatterData.filter(d => d.quadrant === "NOT PRIORITY").map(d => ({ x: d.threat, y: d.confidence })),  backgroundColor: "#4d9fff", pointRadius: 3, pointHoverRadius: 7 },
    ],
  };
  return (
    <div className="bg-[#050505] border border-white/10 p-6 rounded-xl my-8">
      <h4 className="text-white font-mono text-sm mb-4 text-center">Risk Assessment Quadrant Dashboard</h4>
      <div className="h-64 relative">
        <Scatter data={data} options={{ responsive: true, maintainAspectRatio: false, animation: false as any,
          scales: {
            x: { title: { display: true, text: "Threat Score", color: "#888" }, min: 0, max: 100, grid: { color: "rgba(255,255,255,0.05)" } },
            y: { title: { display: true, text: "Confidence", color: "#888" }, min: 0, max: 1.0, grid: { color: "rgba(255,255,255,0.05)" } },
          },
          plugins: { legend: { position: "bottom", labels: { color: "#fff", font: { family: "monospace", size: 10 } } } },
        }} plugins={[quadrantLinesPlugin as any]} />
      </div>
    </div>
  );
};

export default function Paper() {
  const { hash } = useLocation();
  const [isTocOpen, setIsTocOpen] = useState(true);

  // Extract only H2 headings dynamically for the TOC
  const tocEntries = useMemo(() => {
    const slugger = new GithubSlugger();
    const regex = /^##\s+(.+)$/gm;
    let match;
    const entries = [];
    while ((match = regex.exec(paperMarkdown)) !== null) {
      if (match[1].trim() === "TABLE OF CONTENTS") continue;
      const originalTitle = match[1].trim();
      const title = originalTitle.replace(/^\d+\.\s*/, '');
      const id = slugger.slug(originalTitle);
      entries.push({ title, id });
    }
    return entries;
  }, []);

  const tocIds = useMemo(() => tocEntries.map((e) => e.id), [tocEntries]);
  const activeSectionId = useScrollSpy(tocIds, 150);

  useEffect(() => {
    // Smooth scroll for global HTML (implemented safely at component mount)
    document.documentElement.style.scrollBehavior = "smooth";
    
    if (hash) {
      setTimeout(() => {
        const id = hash.replace("#", "");
        const element = document.getElementById(id);
        if (element) {
          const yOffset = -120;
          const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
          window.scrollTo({ top: y, behavior: "smooth" });
        }
      }, 100);
    } else {
      window.scrollTo(0, 0);
    }

    return () => {
      document.documentElement.style.scrollBehavior = "auto";
    };
  }, [hash]);

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([paperMarkdown], { type: "text/markdown" });
    element.href = URL.createObjectURL(file);
    element.download = "DebriSolver_Research_Paper.md";
    document.body.appendChild(element);
    element.click();
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-[#050505] text-[#9ca3af] relative selection:bg-[#00d4ff]/30 selection:text-white pb-24 font-sans leading-relaxed">
      {/* Navbar Minimal */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#050505]/95 backdrop-blur-md border-b border-white/5 px-6 py-4 flex items-center justify-between">
        <Link
          to="/"
          className="flex items-center gap-2 text-[#a0a0a0] hover:text-[#00ff88] transition-colors group"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
          <span className="font-mono text-sm uppercase tracking-widest">
            Back to Project
          </span>
        </Link>
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 bg-[#111111] hover:border-[#00ff88]/50 text-white px-4 py-2 rounded border border-white/10 transition-colors font-mono text-xs uppercase tracking-widest"
        >
          <Download className="w-4 h-4" />
          <span className="hidden sm:inline">Download Markdown</span>
        </button>
      </nav>

      {/* Main Grid Layout */}
      <div className={`max-w-[90rem] mx-auto pt-32 px-6 lg:px-8 relative z-10 animate-slide-up grid grid-cols-1 md:grid-cols-12 gap-6 lg:gap-12 transition-all duration-300`}>
        
        {/* Sticky Sidebar (Table of Contents) */}
        <aside className={`hidden md:block transition-all duration-300 ${isTocOpen ? 'col-span-3' : 'col-span-1'}`}>
          <div className="sticky top-32 max-h-[calc(100vh-10rem)] overflow-y-auto pr-2 hide-scrollbar">
            <button 
              onClick={() => setIsTocOpen(!isTocOpen)}
              className={`flex items-center w-full group transition-all duration-300 mb-6 bg-[#0a0a0a]/80 backdrop-blur-md rounded-xl p-2 border border-white/5 hover:border-[#00ff88]/50 shadow-lg ${isTocOpen ? 'justify-between text-left px-4' : 'justify-center border-[#00ff88]/30'}`}
              title="Toggle Table of Contents"
            >
              {isTocOpen ? (
                <h4 className="text-white font-mono uppercase tracking-widest text-xs border-l-2 border-[#00ff88] pl-3 transition-colors">
                  Contents
                </h4>
              ) : (
                <div className="w-8 h-8 rounded flex items-center justify-center transition-colors">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[#a0a0a0] group-hover:text-[#00ff88] transition-colors">
                    <line x1="8" y1="6" x2="21" y2="6"></line>
                    <line x1="8" y1="12" x2="21" y2="12"></line>
                    <line x1="8" y1="18" x2="21" y2="18"></line>
                    <line x1="3" y1="6" x2="3.01" y2="6"></line>
                    <line x1="3" y1="12" x2="3.01" y2="12"></line>
                    <line x1="3" y1="18" x2="3.01" y2="18"></line>
                  </svg>
                </div>
              )}
              
              {isTocOpen && (
                <div className="text-[#666] group-hover:text-[#00ff88] transition-colors">
                   <ChevronDown className="w-4 h-4" />
                </div>
              )}
            </button>
            
            {isTocOpen && (
              <ul className="space-y-3 bg-[#0a0a0a]/90 backdrop-blur-md border border-white/5 p-4 rounded-xl shadow-[0_10px_30px_rgba(0,0,0,0.8)]">
                {tocEntries.map((entry, index) => (
                  <li key={entry.id}>
                    <a
                      href={`#${entry.id}`}
                      onClick={(e) => {
                        e.preventDefault();
                        const element = document.getElementById(entry.id);
                        if (element) {
                          const yOffset = -120; 
                          const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
                          window.scrollTo({ top: y, behavior: "smooth" });
                          window.history.pushState(null, "", `#${entry.id}`);
                        }
                      }}
                      className={`flex gap-2 font-mono text-[11px] uppercase tracking-wider transition-colors ${
                        activeSectionId === entry.id
                          ? "text-[#00ff88] font-bold"
                          : "text-[#666] hover:text-white"
                      }`}
                    >
                      <span className="opacity-50 min-w-[1.2rem]">{index + 1}.</span>
                      <span>{entry.title}</span>
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        {/* Article Content */}
        <main className={`transition-all duration-300 bg-[#111111] border border-white/5 rounded-2xl p-8 sm:p-12 mb-32 ${isTocOpen ? 'col-span-1 md:col-span-9' : 'col-span-1 md:col-span-11'}`}>
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6 mb-16 border-b border-white/5 pb-10">
            <div className="w-16 h-16 rounded-2xl bg-[#1a1a1a] border border-[#00ff88]/20 flex items-center justify-center shrink-0 shadow-[0_0_30px_rgba(0,255,136,0.1)]">
              <FileText className="w-7 h-7 text-[#00ff88]" />
            </div>
            <div>
              <div className="text-[#00ff88] font-mono text-sm tracking-widest uppercase mb-2">
                DebriSolver Technical Paper
              </div>
              <h1 className="text-3xl sm:text-4xl font-sans tracking-tight font-bold text-[#f3f4f6]">
                Learning Conjunction Dynamics
              </h1>
            </div>
          </div>

          <div className="markdown-body font-sans text-base leading-relaxed space-y-8 prose-strong:text-white prose-strong:font-semibold">
            <Markdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeRaw, rehypeKatex, rehypeSlug]}
              components={{
                h1: ({ node, ...props }) => (
                  <h1 className="hidden" {...props} /> // Hide the original H1 as it's styled in the header
                ),
                h2: ({ node, ...props }) => {
                  // rehypeSlug automatically assigns an id down to the h2
                  // Let's use the ID generated by rehypeSlug, but we need the TOC to match it.
                  return (
                    <h2
                      {...props}
                      className="text-2xl sm:text-3xl font-sans tracking-tight font-bold text-[#f3f4f6] mt-16 mb-8 border-b border-white/5 pb-4 scroll-mt-24"
                    />
                  );
                },
                h3: ({ node, ...props }) => (
                  <h3
                    className="text-xl sm:text-2xl font-sans tracking-tight font-bold text-white mt-12 mb-6 scroll-mt-24"
                    {...props}
                  />
                ),
                a: ({ node, href, children, ...props }) => {
                  if (href && href.startsWith("#")) {
                    return (
                      <a
                        href={href}
                        className="text-[#00ff88] no-underline hover:underline transition-all"
                        onClick={(e) => {
                          e.preventDefault();
                          const targetId = href.replace("#", "");
                          const element = document.getElementById(targetId);
                          if (element) {
                            element.scrollIntoView({ behavior: "smooth" });
                            window.history.pushState(null, "", href);
                          }
                        }}
                        {...props}
                      >
                        {children}
                      </a>
                    );
                  }
                  return (
                    <a
                      href={href}
                      className="text-[#00d4ff] hover:text-white underline transition-colors"
                      target="_blank"
                      rel="noopener noreferrer"
                      {...props}
                    >
                      {children}
                    </a>
                  );
                },
                p: ({ node, ...props }) => (
                  <p className="mb-6 text-[#9ca3af] leading-[1.75]" {...props} />
                ),
                ul: ({ node, ...props }) => (
                  <ul
                    className="list-disc pl-6 mb-6 text-[#9ca3af] space-y-3 marker:text-[#00ff88]/60"
                    {...props}
                  />
                ),
                ol: ({ node, ...props }) => (
                  <ol
                    className="list-decimal pl-6 mb-6 text-[#9ca3af] space-y-3 marker:text-[#00ff88] marker:font-mono"
                    {...props}
                  />
                ),
                li: ({ node, children, ...props }) => (
                  <li className="pl-2" {...props}>
                    {children}
                  </li>
                ),
                strong: ({ node, children, ...props }) => {
                  const text = String(children);
                  if (text.includes("What happened:") || text.includes("Root cause:")) {
                    return <strong className="font-bold text-[#ef4444]" {...props}>{children}</strong>;
                  }
                  if (text.includes("Fix:")) {
                    return <strong className="font-bold text-[#10b981]" {...props}>{children}</strong>;
                  }
                  // Let the prose-strong handle the default color or apply explicitly
                  return <strong className="font-bold text-white" {...props}>{children}</strong>;
                },
                blockquote: ({ node, ...props }) => (
                  <blockquote
                    className="border-l-4 border-[#00ff88] bg-[#00ff88]/5 p-4 rounded-r-lg my-8 text-[#d1d5db] font-serif italic"
                    {...props}
                  />
                ),
                hr: ({ node, ...props }) => (
                  <hr className="border-white/5 my-16" {...props} />
                ),
                table: ({ node, ...props }) => (
                  <div className="overflow-x-auto my-10 border border-white/10 rounded-xl bg-[#0a0a0a]">
                    <table
                      className="w-full text-left border-collapse text-sm whitespace-nowrap"
                      {...props}
                    />
                  </div>
                ),
                th: ({ node, ...props }) => (
                  <th
                    className="bg-[#111111] p-4 text-[#9ca3af] font-mono text-xs uppercase tracking-wider border-b border-white/10"
                    {...props}
                  />
                ),
                td: ({ node, ...props }) => {
                  let className = "p-4 border-b border-white/5 text-[#d1d5db]";
                  // Try to extract text recursively to style dynamically
                  const extractText = (n: any): string => {
                    if (!n) return "";
                    if (n.type === "text") return n.value || "";
                    if (n.children && Array.isArray(n.children)) return n.children.map(extractText).join("");
                    return "";
                  };
                  const text = extractText(node).trim();
                  
                  if (text === "ACT NOW") className += " text-[#ef4444] font-bold";
                  if (text === "WATCH CLOSELY") className += " text-[#f97316] font-bold";
                  if (text === "SAFELY IGNORE") className += " text-[#10b981] font-bold";
                  if (text === "NOT PRIORITY") className += " text-[#3b82f6] font-bold";
                  
                  // Right align numbers optionally
                  const isNumericColumn = /^[-\d.,%Ee+]+$/.test(text);
                  if (text && isNumericColumn) className += " text-right font-mono";
                  return <td className={className} {...props} />;
                },
                pre: ({ node, children, ...props }) => {
                  const firstChild = node?.children?.[0] as any;
                  let isCustomChart = false;
                  if (firstChild && firstChild.tagName === 'code') {
                    const className = firstChild.properties?.className || [];
                    const match = className.find((c: string) => c.startsWith('language-'));
                    const customChartNames = [
                      "language-losschart", "language-quadrantchart", "language-preprocessingdiagram",
                      "language-pipelinediagram", "language-uncertaintyplot", "language-grucelldiagram",
                      "language-threatdistributionchart", "language-confidencedistributionchart",
                      "language-threatvsactualpcchart", "language-confidencevscdmschart",
                      "language-maecurvechart", "language-pcpredictionerrorchart", "language-teamcards"
                    ];
                    if (match && customChartNames.includes(match)) {
                      isCustomChart = true;
                    }
                  }
                  
                  if (isCustomChart) {
                    return <>{children}</>;
                  }

                  return (
                    <pre
                      className="bg-[#1a1a1a] p-6 rounded-xl overflow-x-auto my-8 border border-white/10 shadow-inner"
                      {...props}
                    >
                      {children}
                    </pre>
                  );
                },
                code(props: any) {
                  const { children, className, node, inline, ...rest } = props;
                  const match = /language-(\w+)/.exec(className || "");
                  
                  if (match && match[1] === "losschart") return <TrainingLossChart />;
                  if (match && match[1] === "quadrantchart") return <QuadrantChart />;
                  if (match && match[1] === "preprocessingdiagram") return <PreprocessingDiagram />;
                  if (match && match[1] === "pipelinediagram") return <PipelineDiagram />;
                  if (match && match[1] === "uncertaintyplot") return <UncertaintyPlot />;
                  if (match && match[1] === "grucelldiagram") return <GRUCellDiagram />;
                  if (match && match[1] === "threatdistributionchart") return <ThreatDistributionChart />;
                  if (match && match[1] === "confidencedistributionchart") return <ConfidenceDistributionChart />;
                  if (match && match[1] === "threatvsactualpcchart") return <ThreatVsPcChart />;
                  if (match && match[1] === "confidencevscdmschart") return <ConfidenceVsCdmsChart />;
                  if (match && match[1] === "maecurvechart") return <MaecurveChart />;
                  if (match && match[1] === "pcpredictionerrorchart") return <PcPredictionErrorChart />;
                  if (match && match[1] === "teamcards") return <TeamCards />;
                  
                  // For multi-line code blocks
                  if (match) {
                    return (
                      <code className={`${className} font-mono text-sm text-[#d1d5db]`} {...rest}>
                        {children}
                      </code>
                    );
                  }
                  
                  // For inline code
                  return (
                    <code
                      className="bg-[#1a1a1a] px-1.5 py-0.5 rounded text-[#d1d5db] font-mono text-[0.9em] border border-white/5"
                      {...rest}
                    >
                      {children}
                    </code>
                  );
                },
              }}
            >
              {paperMarkdown}
            </Markdown>
          </div>
        </main>
      </div>

      {/* Floating Back to Top Button */}
      <button
        onClick={scrollToTop}
        className="fixed bottom-8 right-8 w-12 h-12 bg-[#111111] hover:bg-[#1a1a1a] text-white border border-white/10 hover:border-[#00ff88]/50 rounded-full flex items-center justify-center shadow-lg transition-all group z-50"
        aria-label="Back to top"
      >
        <ArrowUp className="w-5 h-5 group-hover:-translate-y-1 transition-transform" />
      </button>
    </div>
  );
}


