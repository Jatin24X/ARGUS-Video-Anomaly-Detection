# Design System: ARGUS Video Anomaly Detection Platform

## 1. Visual Theme & Atmosphere
The ARGUS dashboard is designed as a premium **Security Operations Command Center**. The atmosphere is characterized by a hybrid of **brutalist data density** and **tactical telemetry depth**, prioritizing functional clarity, high information readouts, and military control room aesthetics.
- **Density:** Cockpit Dense (8/10) — maximizes structural space for metrics, real-time logs, and timelines.
- **Variance:** Offset Asymmetric (5/10) — grids align with structured columns but balance varied widths (col-span-4, col-span-8).
- **Motion:** Weighted Spring Physics (6/10) — framer-motion springs with weight, inertia, and micro-hover lifts.

## 2. Color Palette & Roles
- **Command Slate** (`#03070b`) — Primary background canvas
- **Tactical Panel** (`rgba(13, 18, 23, 0.85)`) — Card and panel background surface
- **Ink Primary** (`#f0f6fc`) — Primary text and high-contrast labels
- **Muted Steel** (`#8b9eb0`) — Secondary descriptions, captions, and table headers
- **Whisper Border** (`rgba(140, 180, 200, 0.09)`) — Default card border and dividers
- **Telemetry Cyan** (`#29d3ff`) — Active monitoring status, timeline values, and search accents
- **Incident Red** (`#ff5a60`) — Critical anomalies, active alarm pulses, and threshold warnings
- **Warning Amber** (`#ffb25f`) — Warning logs, warning badges, and paused feed states
- **Calibrated Mint** (`#65f0b6`) — Healthy status ticks, pipeline completes, and model success badges

## 3. Typography Rules
- **Display/Headlines:** `Space Grotesk` — Track-tight, weight-driven hierarchy, uppercase labels for widgets.
- **Body Text:** `Manrope` — Highly readable sans-serif for descriptions, accordion Q&As, and body text.
- **Telemetry/Mono:** `JetBrains Mono` — For all numerical counts, timestamps, chart intervals, live exception tables, and system execution logs.
- **Banned:** `Inter` is strictly forbidden. Emojis and generic serif fonts (e.g. Times New Roman) are banned.

## 4. Component Stylings
* **Bento Cards (Double-Bezel Nested Architecture):**
  - **Outer Frame (`.bento-card-shell`):** `border: 1px solid rgba(255, 255, 255, 0.04); background: rgba(255, 255, 255, 0.02); padding: 6px; border-radius: 28px;`
  - **Inner Core (`.bento-card`):** `border: 1px solid rgba(255, 255, 255, 0.05); background: linear-gradient(135deg, rgba(10, 16, 26, 0.6) 0%, rgba(5, 8, 12, 0.8) 100%); padding: 24px; border-radius: 22px;`
  - **Cursor Interaction:** On hover, a radial cursor glow overlays the outer shell, and the panel lifts smoothly (`translateY(-4px)`).
* **Buttons:** Flat telemetry buttons. On active click, translates `-1px` vertically.
* **Badges:** Small inline status indicators with mono tags and colored pulse dots (Cyan for Active, Amber for Warning, Mint for Success).
* **Loaders:** Skeleton blocks matching exact layout grid proportions. Bouncing spinners are banned.

## 5. Layout Principles
- **Grid Structure:** 12-column bento grid collapsing intelligently to 6-column on tablets, and single-column on small screens.
- **Spacing:** Structured margins (`gap: 24px` on desktop) utilizing flex/grid spacing rather than percentage hacks.
- **Containment:** Layout wrapper constrained to a clean maximum width (`max-width: 1400px`) centered on viewport.

## 6. Motion & Interaction
- **Spring Defaults:** Frame rates animated at weighted springs: `stiffness: 170`, `damping: 19` for timelines, tooltips, and slider interactions.
- **Staggered Orchestration:** Card entrances stagger progressively based on card index (`--card-index * 80ms`).
- **Performance:** Hardware-accelerated animations restricted to `transform` and `opacity` to maintain smooth 60fps on telemetry redraws.

## 7. Anti-Patterns (Banned)
- No emojis inside command telemetry.
- No purple or pink neon gradients.
- No plain or generic box layouts (everything uses nested double-bezels).
- No fake numbers or mock labels; all telemetry mirrors active engine parameters.
- No paragraph text in headers.
- No centered hero layout in dashboard pages.
