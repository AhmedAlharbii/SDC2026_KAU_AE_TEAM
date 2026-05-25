import Hero from '../components/Hero';
import Problem from '../components/Problem';
import Solution from '../components/Solution';
import Architecture from '../components/Architecture';
import Dashboard from '../components/Dashboard';
import Credits from '../components/Credits';

export default function Home() {
  return (
    <>
      {/* Global Background Effects matching Heimdall radar/space vibe */}
      <div className="grid-bg"></div>
      
      <main className="flex-1 w-full flex flex-col z-10 relative">
        <Hero />
        <Problem />
        <Solution />
        <Architecture />
        <Dashboard />
        <Credits />
      </main>
    </>
  );
}
