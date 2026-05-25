export default function GRUCellDiagram() {
  return (
    <svg viewBox="-20 40 460 260" className="w-full h-auto drop-shadow-2xl" style={{ fontFamily: 'monospace' }}>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#888888" />
        </marker>
        <marker id="arrow-white" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ffffff" />
        </marker>
      </defs>

      {/* Main cell background */}
      <rect x="50" y="50" width="300" height="200" rx="10" fill="#030303" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
      <text x="60" y="70" fill="#555" fontSize="12" fontWeight="bold">GRU CELL</text>

      {/* Inputs */}
      <text x="10" y="154" fill="#ffffff" fontSize="16" fontWeight="bold" textAnchor="middle">h(t-1)</text>
      <line x1="30" y1="150" x2="50" y2="150" stroke="#ffffff" strokeWidth="2" markerEnd="url(#arrow-white)" />
      
      <text x="150" y="285" fill="#ffffff" fontSize="16" fontWeight="bold" textAnchor="middle">x(t)</text>
      <line x1="150" y1="265" x2="150" y2="250" stroke="#ffffff" strokeWidth="2" markerEnd="url(#arrow-white)" />

      {/* Gates */}
      <rect x="100" y="180" width="40" height="25" rx="3" fill="#111" stroke="#333" strokeWidth="1" />
      <text x="120" y="197" fill="#00ff88" fontSize="12" fontWeight="bold" textAnchor="middle">σ (r)</text>

      <rect x="170" y="180" width="40" height="25" rx="3" fill="#111" stroke="#333" strokeWidth="1" />
      <text x="190" y="197" fill="#ff5a5a" fontSize="12" fontWeight="bold" textAnchor="middle">σ (z)</text>

      {/* Tanh */}
      <rect x="240" y="180" width="40" height="25" rx="3" fill="#111" stroke="#333" strokeWidth="1" />
      <text x="260" y="197" fill="#ffbd2e" fontSize="12" fontWeight="bold" textAnchor="middle">tanh</text>

      {/* Operators */}
      <circle cx="120" cy="110" r="10" fill="#111" stroke="#666" strokeWidth="1" />
      <text x="120" y="114" fill="#fff" fontSize="12" textAnchor="middle">X</text>

      <circle cx="190" cy="90" r="10" fill="#111" stroke="#666" strokeWidth="1" />
      <text x="190" y="94" fill="#fff" fontSize="12" textAnchor="middle">X</text>

      <circle cx="260" cy="110" r="10" fill="#111" stroke="#666" strokeWidth="1" />
      <text x="260" y="114" fill="#fff" fontSize="12" textAnchor="middle">X</text>

      <circle cx="300" cy="150" r="10" fill="#111" stroke="#666" strokeWidth="1" />
      <text x="300" y="154" fill="#fff" fontSize="12" textAnchor="middle">+</text>

      {/* Tanh block candidate internal */}
      <circle cx="260" cy="140" r="12" fill="none" />

      {/* Flow Lines */}
      {/* x(t) routing */}
      <path d="M 150 250 L 150 220" fill="none" stroke="#888" strokeWidth="2" />
      <path d="M 150 220 L 120 220 L 120 205" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />
      <path d="M 150 220 L 190 220 L 190 205" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />
      <path d="M 150 220 L 260 220 L 260 205" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* h(t-1) routing */}
      <path d="M 50 150 L 70 150 L 70 235 L 120 235 L 120 205" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />
      <path d="M 70 235 L 190 235 L 190 205" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* r gate to multi */}
      <path d="M 120 180 L 120 120" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />
      {/* h(t-1) to r multi */}
      <path d="M 70 150 L 70 110 L 110 110" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* r multi to tanh */}
      <path d="M 130 110 L 250 110 L 250 180" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* z gate routing */}
      <path d="M 190 180 L 190 100" fill="none" stroke="#ff5a5a" strokeWidth="2" markerEnd="url(#arrow)" />
      <path d="M 190 170 L 210 170 L 210 70 L 260 70 L 260 100" fill="none" stroke="#ff5a5a" strokeDasharray="4,4" strokeWidth="2" markerEnd="url(#arrow)" />
      <text x="210" y="58" fill="#ff5a5a" fontSize="12" fontWeight="bold">1 - z(t)</text>

      {/* h(t-1) straight across top */}
      <path d="M 70 150 L 70 90 L 180 90" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />
      
      {/* 1-z mult to add */}
      <path d="M 200 90 L 300 90 L 300 140" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* tanh to mult */}
      <path d="M 260 180 L 260 120" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />
      {/* z mult to add */}
      <path d="M 270 110 L 300 110 L 300 140" fill="none" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* Add to Output */}
      <line x1="310" y1="150" x2="350" y2="150" stroke="#888" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="350" y1="150" x2="385" y2="150" stroke="#ffffff" strokeWidth="3" markerEnd="url(#arrow-white)" />
      <text x="405" y="156" fill="#ffffff" fontSize="16" fontWeight="bold" textAnchor="start">h(t)</text>
    </svg>
  );
}
