import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="border-t border-[#ffffff]/[0.08] bg-[#030303] pt-16 pb-8 px-[clamp(1.5rem,5vw,4rem)]">
      <div className="max-w-[1400px] mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-8 md:space-y-0 mb-12">
          <div className="flex flex-col space-y-4">

            <p className="font-mono text-xs text-[#808080] max-w-sm">
              Learning conjunction dynamics through a self-supervised approach. 
              Estimating satellite collision risks without direct labels.
            </p>
          </div>

          <div className="flex flex-col md:items-end space-y-4">
            <nav className="flex space-x-6">
              <Link to="/" className="font-mono text-xs tracking-widest text-[#808080] hover:text-[#00ff88] uppercase transition-colors">Home</Link>
              <Link to="/paper" className="font-mono text-xs tracking-widest text-[#808080] hover:text-[#00ff88] uppercase transition-colors">Paper</Link>
              <a href="https://github.com/AhmedAlharbii/SDC2026_KAU_AE_TEAM" target="_blank" rel="noopener noreferrer" className="font-mono text-xs tracking-widest text-[#808080] hover:text-[#00ff88] uppercase transition-colors">GitHub</a>
            </nav>
            <p className="font-mono text-[10px] tracking-widest text-[#555] uppercase">
              SDC 2026 • KAU AE TEAM
            </p>
          </div>
        </div>
        
        <div className="border-t border-[#ffffff]/[0.05] pt-8 flex flex-col md:flex-row justify-between items-center text-[#555] font-mono text-[10px] tracking-[0.2em] uppercase">
          <p>© {new Date().getFullYear()} Space Track. All rights reserved.</p>
          <p className="mt-4 md:mt-0">Debris Risk Assessment Platform</p>
        </div>
      </div>
    </footer>
  );
}
