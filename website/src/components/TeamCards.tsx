import React from 'react';
import { Linkedin } from 'lucide-react';

const team = [
  { name: 'Ahmad Alharbi', role: 'Team Lead & Lead Developer', linkedin: 'ahmed-alharbi-973b63246', link: 'https://www.linkedin.com/in/ahmed-alharbi-973b63246/' },
  { name: 'Abdulelah Mojelad', role: 'AI Research & Development', linkedin: 'abdulellah-mojalled', link: 'https://www.linkedin.com/in/abdulellah-mojalled/' },
  { name: 'Hamzah Alharbi', role: 'Research & Development', linkedin: 'hamzah-alharbi-00b18133a', link: 'https://www.linkedin.com/in/hamzah-alharbi-00b18133a/' },
  { name: 'Khalid Alsadoon', role: 'Research & Development', linkedin: 'khalid-alsadoon-a95802242', link: 'https://www.linkedin.com/in/khalid-alsadoon-a95802242/' },
  { name: 'Mohamedhakim Hassan', role: 'Research & Development', linkedin: 'mohamed-hassan-aero', link: 'https://www.linkedin.com/in/mohamed-hassan-aero/' }
];

export default function TeamCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 my-10">
      {team.map((member, i) => (
        <div key={i} className="bg-[#111111] border border-white/10 rounded-2xl p-6 hover:border-[#00ff88]/50 transition-colors group relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-[#00ff88]/5 rounded-bl-full translate-x-16 -translate-y-16 group-hover:bg-[#00ff88]/10 transition-colors"></div>
          
          <div className="flex flex-col h-full relative z-10">
            <div className="w-12 h-12 rounded-full bg-[#1a1a1a] border border-[#00ff88]/20 flex items-center justify-center font-mono text-[#00ff88] text-xl mb-4 group-hover:scale-110 transition-transform">
              {member.name.charAt(0)}
            </div>
            
            <h4 className="text-white font-semibold text-lg tracking-tight mb-1">{member.name}</h4>
            <p className="text-[#a0a0a0] text-sm mb-6 flex-grow">{member.role}</p>
            
            <a 
              href={member.link} 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-[#666] hover:text-[#00ff88] transition-colors mt-auto w-fit"
            >
              <Linkedin className="w-4 h-4" />
              <span className="font-mono text-xs truncate max-w-[200px]">{member.linkedin}</span>
            </a>
          </div>
        </div>
      ))}
    </div>
  );
}
