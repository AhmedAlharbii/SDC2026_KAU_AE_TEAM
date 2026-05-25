import { Line } from 'react-chartjs-2';

export default function UncertaintyPlot() {
  const data = {
    labels: ['T-5', 'T-4', 'T-3', 'T-2', 'T-1', 'Target', 'T+1'],
    datasets: [
      {
        label: 'Mean Prediction (Pc)',
        data: [1e-7, 5e-7, 2e-6, 1e-5, 8e-5, 2e-4, 5e-4],
        borderColor: '#00ff88',
        backgroundColor: 'transparent',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: '#00ff88'
      },
      {
        label: 'Upper Bound (95% CI)',
        data: [1e-6, 8e-6, 3e-5, 1e-4, 5e-4, 1e-3, 3e-3],
        borderColor: 'rgba(0, 255, 136, 0.2)',
        backgroundColor: 'rgba(0, 255, 136, 0.1)',
        tension: 0.4,
        pointRadius: 0,
        fill: '+1'
      },
      {
        label: 'Lower Bound (95% CI)',
        data: [1e-8, 1e-8, 5e-7, 2e-6, 1e-5, 3e-5, 8e-5],
        borderColor: 'rgba(0, 255, 136, 0.2)',
        backgroundColor: 'transparent',
        tension: 0.4,
        pointRadius: 0,
        fill: false
      }
    ]
  };

  return (
    <div className="bg-[#050505] border border-white/10 p-6 rounded-xl my-8">
      <h4 className="text-white font-mono text-sm mb-2 tracking-widest uppercase">Monte Carlo Dropout Uncertainty</h4>
      <p className="text-[#a0a0a0] font-sans text-sm mb-6 leading-relaxed">
        Illustrating 50 forward sampling passes. The expanding envelope shows how prediction uncertainty grows as the model extrapolates further into the future or encounters out-of-distribution tracking noise.
      </p>
      <div className="h-64">
        <Line 
          data={data} 
          options={{
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { 
                type: 'logarithmic',
                grid: { color: 'rgba(255,255,255,0.05)' }, 
                border: { dash: [4, 4] }, 
                ticks: { color: '#888' },
                title: { display: true, text: 'Collision Probability (Pc)', color: '#fff' }
              },
              x: { 
                grid: { color: 'rgba(255,255,255,0.05)' }, 
                border: { dash: [4, 4] }, 
                ticks: { color: '#888' },
                title: { display: true, text: 'Conjunction Timeline', color: '#fff' }
              }
            },
            plugins: { 
              legend: { labels: { color: '#fff', font: { family: 'monospace', size: 10 } } },
              tooltip: { mode: 'index', intersect: false }
            }
          }} 
        />
      </div>
    </div>
  );
}
