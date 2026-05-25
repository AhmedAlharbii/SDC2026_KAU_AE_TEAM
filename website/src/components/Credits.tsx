import { Linkedin } from 'lucide-react';
import { motion } from 'motion/react';

export default function Credits() {
  const team = [
    { name: "Ahmad Alharbi", role: "Team Lead", linkedin: "https://www.linkedin.com/in/ahmed-alharbi-973b63246/" },
    { name: "Abdulelah Mojelad", role: "AI Lead", linkedin: "https://www.linkedin.com/in/abdulellah-mojalled/" },
    { name: "Hamzah Alharbi", role: "R&D (Aerospace)", linkedin: "https://www.linkedin.com/in/hamzah-alharbi-00b18133a/" },
    { name: "Khalid Alsadoon", role: "R&D (Aerospace)", linkedin: "https://www.linkedin.com/in/khalid-alsadoon-a95802242/" },
    { name: "Mohamedhakim Hassan", role: "R&D (Aerospace)", linkedin: "https://www.linkedin.com/in/mohamed-hassan-aero/" },
  ];

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
    <footer id="paper" className="pt-[clamp(4rem,10vw,8rem)] pb-[clamp(3rem,8vw,6rem)] px-[clamp(1.5rem,5vw,4rem)] w-full font-mono text-xs border-t border-[#ffffff]/[0.08] text-[#808080] overflow-hidden animate-slide-up">
      <motion.div 
        className="max-w-4xl mx-auto space-y-12"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={staggerContainer}
      >
        <div className="w-full h-px bg-gradient-to-r from-transparent via-[rgba(255,255,255,0.1)] to-transparent"></div>

        <motion.div variants={fadeInUp} className="text-center tracking-widest uppercase mb-16 text-white font-bold">
          KAU AEROSPACE ENGINEERING &bull; SDC2026
        </motion.div>

        <motion.div variants={staggerContainer} className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-4 max-w-2xl mx-auto">
          {team.map((member, i) => (
            <motion.div variants={fadeInUp} key={i} className={`flex justify-between items-center border-b border-[#ffffff]/[0.08] pb-2 p-2 hover:bg-white/[0.02] transition-colors ${i === 4 ? 'col-span-1 md:col-span-2 max-w-sm mx-auto w-full' : ''}`}>
              <div className="flex items-center gap-3">
                <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="text-[#0077b5] hover:text-[#00a0dc] transition-colors" title={`View ${member.name} on LinkedIn`}>
                   <Linkedin size={18} />
                </a>
                <span className="text-white text-sm">{member.name}</span>
              </div>
              <span className="text-[#a0a0a0]">{member.role}</span>
            </motion.div>
          ))}
        </motion.div>

        <div className="w-full h-px bg-gradient-to-r from-transparent via-[rgba(255,255,255,0.05)] to-transparent my-12"></div>

        <motion.div variants={fadeInUp} className="text-center space-y-6">
          <div className="tracking-[0.2em] uppercase text-[10px] text-[#808080]">SUPPORTED BY</div>
          <div className="flex flex-wrap justify-center gap-6 items-center tracking-widest text-[#ffffff]">
             <span>&#9670; Saudi Space Agency</span>
             <span>&#9670; ALDORIA</span>
             <span>&#9670; King Abdulaziz University</span>
          </div>
        </motion.div>

        <div className="w-full h-px bg-gradient-to-r from-transparent via-[rgba(255,255,255,0.05)] to-transparent my-12"></div>

        <div className="text-center text-[10px] tracking-widest pt-12 mt-12 border-t border-[#ffffff]/[0.08]">
            Built for safer space operations &bull; KAU &bull; 2026
        </div>
      </motion.div>
    </footer>
  );
}
