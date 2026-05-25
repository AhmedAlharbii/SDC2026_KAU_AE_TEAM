# DebriSolver Showcase Website — Implementation Plan (v5 — Final)

## Goal

Build a **cinematic, HUD-styled showcase website** with Heimdall Space aesthetics, Grafana-like dashboard panels, and NASA Eyes feel. **100% code — zero images of any kind.** Every visual is HTML + CSS + JS + Chart.js.

---

## Decisions

| Decision | Answer |
|----------|--------|
| **Location** | `SDC2026_KAU_AE_TEAM/website/` |
| **Images** | ❌ **NONE. Zero.** No PNGs, no AI-generated, no external files. Everything is code. |
| **Backgrounds** | CSS gradients + radial glows + particle field (JS-spawned divs) |
| **Charts** | Chart.js (CDN) |
| **Diagrams** | HTML + CSS (flexbox/grid blocks with borders) |
| **Gauges** | CSS `conic-gradient` arcs |
| **Team** | Minimal text list, no photos |
| **Language** | English |
| **Hosting** | Vercel-ready static |

---

## How Every Visual Is Built (Pure Code)

| Visual | Technique |
|--------|-----------|
| Space background | CSS `radial-gradient` layers (dark blue core → black) + scattered star dots via CSS `box-shadow` on a single pseudo-element |
| Debris particle field | JS creates 40–60 `<div>` elements with randomized CSS `animation` (drift + opacity pulse) |
| HUD corner brackets | 4 `<div>` elements with `border-left + border-top` combos, `position: fixed` |
| Crosshairs | 5 small absolute `<div>` dots with opacity pulse keyframes |
| Training loss curve | Chart.js line chart, gradient fill, dark theme |
| Quadrant scatter plot | Chart.js scatter, colored by quadrant, custom tooltips |
| Donut chart | Chart.js doughnut with center text plugin |
| Bar charts | Chart.js bar with custom colors |
| Model architecture | HTML `<div>` blocks in a flex column, colored left-borders, CSS `::after` connecting lines |
| Pipeline flow | HTML flex row of node cards with CSS dashed-border connectors |
| Gauges | CSS `conic-gradient` semicircles with overlaid center text |
| Progress bars | CSS `width: N%` bars inside dark containers |
| Tables | HTML `<table>` with styled rows, colored badges, monospace values |
| Status indicators | CSS `border-radius: 50%` dots with `box-shadow` glow + pulse animation |
| Telemetry scroll | CSS `translateY` infinite animation on a column of monospace text |

---

## Design Tokens

```css
:root {
  --bg-void: #000000;
  --bg-primary: #050505;
  --bg-secondary: #0a0a0a;
  --bg-elevated: #141414;
  --bg-surface: #1a1a1a;

  --text-primary: #ffffff;
  --text-secondary: #a0a0a0;
  --text-tertiary: #808080;
  --text-muted: #555555;

  --border-faint: rgba(255, 255, 255, 0.04);
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-default: rgba(255, 255, 255, 0.12);

  --status-danger: #ff3366;
  --status-warning: #ff8c00;
  --status-success: #00ff88;
  --status-info: #00d4ff;

  --font-mono: "JetBrains Mono", "SF Mono", "Consolas", monospace;
  --font-display: "Barlow", "Inter", -apple-system, sans-serif;
}
```

---

## 7 Sections

### 1. 🌌 HERO
- **Background**: CSS `radial-gradient(ellipse at 50% 120%, #0a1628 0%, #000 70%)` + star field via `box-shadow` (200+ tiny white dots on a single `::before`) + JS particle field (40 drifting divs)
- HUD corners (fixed brackets), crosshairs (pulsing dots)
- Monospace label: `· SYSTEM STATUS: ACTIVE ·`
- Barlow headline: `EARTH'S ORBIT IS FILLING UP WITH JUNK`
- Monospace subtitle with stats
- White CTA button
- Readout bar with `dashboard_summary.json` stats
- Scroll hint chevron

### 2. ⚠️ THE PROBLEM
- Counter cards (monospace numbers, count-up animation)
- Iridium-Cosmos callout (red-bordered panel)
- "5 Failures" sequential reveal cards
- Punch line in Barlow display

### 3. 💡 OUR SOLUTION
- Competition badge bar (monospace, dot-separated)
- CDM timeline (CSS circles + dashed connecting line + animated arrow)
- Four-quadrant 2×2 grid (colored border accents, hover glow)
- Self-supervised explanation text

### 4. 🧠 ARCHITECTURE
- Model diagram (HTML/CSS vertical block flow with `::after` connectors)
- Decision cards (horizontal scroll strip)
- Pipeline nodes (flex row + dashed CSS connectors)
- Hyperparameter table (dark styled HTML)

### 5. 📊 RESULTS
- Grafana-style panel grid:
  - Training loss curve (Chart.js, 151 epochs, dual lines)
  - Quadrant donut (Chart.js)
  - Metric readout cards (HUD-style counters)
  - Confidence calibration bars (Chart.js)

### 6. 🖥️ LIVE DASHBOARD
- `● LIVE` pulsing indicator + mission control header
- CSS arc gauges (conic-gradient) for threat/confidence
- Chart.js interactive scatter (200 events, tooltips with object names)
- Styled HTML table (10 high-priority events, progress bars, badges)
- Expandable event detail on click

### 7. 👥 CREDITS
- Clean monospace text list (names + roles)
- Acknowledgment badges (SSA, ALDORIA, KAU)
- Citation code block with copy button
- Footer

---

## Data → JS Embedding

| Source | Feeds |
|--------|-------|
| `dashboard_summary.json` | Hero readout, donut chart, gauges, priority table |
| `event_dashboard.csv` (top 200 rows) | Scatter plot, threat histogram |
| `training_history.csv` (151 epochs) | Training loss curve |
| `offline_evaluation_summary.json` | Metric cards |
| `calibration_bins.csv` | Calibration bar chart |

All pre-parsed and hardcoded as JS objects/arrays in `script.js`.

---

## File Structure

```
SDC2026_KAU_AE_TEAM/
└── website/
    ├── index.html
    ├── styles.css
    └── script.js
```

Three files. That's it. No assets folder. No images. Pure code.

---

## Verification

- Open `index.html` in Chrome — all 7 sections render
- HUD frame animating (corners, crosshairs, particles)
- All Chart.js charts load with real data
- Scatter plot tooltips show object names
- CSS gauges render correctly
- Scroll animations trigger
- Responsive at 1920, 1440, 1024, 768
- Vercel deploy: drag `website/` folder
