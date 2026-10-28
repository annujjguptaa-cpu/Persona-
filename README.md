# Persona

**Synthetic Audience Message Spread Simulator**

Persona is a local web application that simulates how any candidate message (alert, campaign, rumor, phishing lure, or marketing broadcast) spreads through a diverse synthetic audience.

It combines:
1. **Local LLM Intelligence (Ollama)** for behavioral, deception, and psychological feature extraction.
2. **Behavioral Archetypes** representing demographic cohorts, cognitive biases, and susceptibility profiles.
3. **Graph Diffusion Engine (NetworkX)** modeling probabilistic multi-wave information cascades.
4. **Scoring & Insights** producing actionable defense verdicts, reach analytics, and vulnerability breakdowns.

---

## Project Structure

```
persona/
├── backend/
│   ├── main.py            # FastAPI entrypoint, GET /health, static mount
│   ├── config.py          # Environment settings (OLLAMA_URL, MODEL, SEED, etc.)
│   ├── models.py          # Strict Pydantic v2 domain models
│   ├── domain_packs.py    # Domain-specific presets and audience definitions
│   ├── archetypes.py      # Behavioral archetype definitions and trait distributions
│   ├── population.py      # Population sampling and agent synthesis
│   ├── network.py         # Social graph generation (scale-free, small-world)
│   ├── analyzer.py        # Message feature extraction (urgency, cues, flags)
│   ├── reactions.py       # Archetype response probability matrix
│   ├── engine.py          # Multi-wave graph cascade simulation engine
│   ├── scoring.py         # Virality, hazard, and conversion scoring
│   ├── insights.py        # Strategic defense recommendations & narrative verdict
│   ├── llm.py             # Local Ollama client & structured prompt handling
│   ├── cache.py           # Disk/memory caching for simulation runs
│   └── safety.py          # Input validation, sanitization & guardrails
├── frontend/
│   └── index.html         # Modern dark-mode dashboard & telemetry UI
├── data/
│   ├── cache/             # Transient simulation cache (.gitignored)
│   └── presets/           # Reusable scenario presets
├── tests/
│   └── test_models.py     # Pytest test suite for models and health check
├── requirements.txt       # Dependencies (fastapi, uvicorn, networkx, etc.)
└── README.md
```

---

## Core Models (`backend/models.py`)

All models are built with **Pydantic v2**. All probabilities and intensity rates are bounded between `0.0` and `1.0` using `Field(ge=0, le=1)`.

- **`Archetype`**: `id`, `label`, `description`, `share_of_population` (0.0 - 1.0), `traits: dict[str, float]`
- **`Person`**: `id`, `name`, `archetype_id`, `age` (>= 0), `language`, `area`, `traits: dict[str, float]`
- **`MessageFeatures`**: `is_suspicious`, `urgency` (0-1), `emotional_pull` (0-1), `authority_claim` (0-1), `call_to_action` (bool/0-1), `has_link`, `language`, `channel_guess`, `red_flags`, `legit_signals`, `topics`
- **`Reaction`**: `archetype_id`, `ignore` (0-1), `notice` (0-1), `engage` (0-1), `act` (0-1), `share` (0-1), `report` (0-1), `top_trigger`, `reason`
- **`WaveState`**: `wave`, `reached_ids`, `newly_reached`, `cumulative_reached`, `reactions`, `shares`, `acts`, `reports`, `details`
- **`SimulationResult`**: `waves`, `edges`, `totals`, `group_stats`, `verdict`, `insights`

---

## Domain Packs (`backend/domain_packs.py`)

A domain pack tells the simulator what a message means in a specific context:
- `scam_fraud`: Harmful. Act = "pays or clicks". Tiers: Contained / Niche hit / Wide spread / Outbreak.
- `health_campaign`: Beneficial. Act = "takes the health action". Tiers: Ignored / Niche reach / Strong reach / Breakout.
- `product_marketing`: Beneficial. Act = "buys or signs up". Tiers: Flop / Niche hit / Strong hit / Viral.
- `civic_policy`: Beneficial. Act = "complies or enrols". Tiers: Unnoticed / Patchy reach / Good reach / Universal.
- `misinformation`: Harmful. Act = "believes and reshares". Tiers: Fizzles out / Local rumour / Widely believed / Viral falsehood.

Each pack features:
- Exactly 4 verdict labels (tiered low to high spread)
- Exactly 3 sample messages (including at least one legitimate benchmark)
- Exactly 3 calibrated audience presets
- Universal trait weights and feature weights mapped in logit space
- `allow_message_improvement = False` for harmful packs, `True` for beneficial packs

Endpoints:
- `GET /packs`: Returns public views of all packs without internal weights.

---

## Audience Archetypes (`backend/archetypes.py`)

1. **Hand-Crafted Presets (`PRESETS`)**:
   Contains 9 Indian contextual presets, each with 10 hand-curated `Archetype` objects:
   - `urban_seniors` (Pensioners, WhatsApp uncles, devotional matriarchs, RWA secretaries...)
   - `college_students` (Intern hustlers, exam crammers, meme influencers, CS skeptics...)
   - `rural_farmers` (Progressive growers, marginal smallholders, mandi dealers, ryots...)
   - `working_parents` (Dual-career parents, class group admins, pediatric researchers...)
   - `small_shopkeepers` (Kirana veterans, soundbox traders, kiosk vendors, GST-anxious merchants...)
   - `community_health_workers` (ASHA didis, Anganwadi workers, PHC nurses, counselors...)
   - `gig_workers` (Quick-commerce riders, cab drivers, daily wagers, night couriers...)
   - `urban_professionals` (Team leads, fintech adopters, security skeptics, consultants...)
   - `homemakers` (Budget CFOs, family network hubs, Ayurvedic enthusiasts, ward volunteers...)

   *Guarantees*: All traits are bounded in `[0.0, 1.0]`, shares sum to `1.0`, and each preset includes at least 2 skeptical (`skepticism > 0.6`) and 2 vulnerable (`digital_literacy < 0.4`) segments.

2. **Custom Archetype Generation (`generate_archetypes`)**:
   - Calls Ollama LLM with structured JSON schema.
   - Enforces trait clamping, share renormalization, and diversity validation ($\ge 2$ skeptics, $\ge 2$ vulnerable).
   - Retries up to 2 times upon invalid outputs before gracefully falling back to the closest hand-crafted preset.
   - Cached by `hash(description)` in memory and disk (`data/cache/`).

Endpoints:
- `GET /presets`: Returns all hand-curated presets and their segment distributions.
- `POST /archetypes/generate`: Synthesizes or retrieves custom audience archetypes.

---

## Population & Social Graph (`backend/population.py` & `backend/network.py`)

### 1. Population Sampling (`generate_population`)
- Samples $N$ individuals from archetype distributions with reproducible `seed`.
- Applies **Gaussian jitter ($SD = 0.08$)** clipped to $[0.0, 1.0]$ to trait means.
- Assigns culturally authentic Indian names categorized by language (`hi`, `mr`, `ta`, `te`, `bn`, `kn`, `en`) and area (`metro`, `suburban`, `rural`).
- Bounds ages to realistic ranges according to archetype profiles.

### 2. Network Topology (`build_network`)
- **Homophily**: Higher connection probabilities for agents sharing the same archetype, geography, or language.
- **Household Clusters**: People grouped into family units of 3-5 members connected by high-weight `family` edges.
- **Community Hubs**: 5-6 designated nodes (leaders, shopkeepers, doctors, influencers, admins) with elevated reach.
- **Channel Dynamics**:
  - `whatsapp`: Dense small-world clusters with group chat cliques ($C > 0.15$).
  - `sms`: Sparse point-to-point connections with lower clustering.
  - `social_feed`: Scale-free star hubs reaching up to 50%+ of the audience.
  - `call`: Near-zero forwarding, mostly dyadic family ties.
- **Guaranteed Connectivity**: Giant component contains $> 90\%$ of nodes.

Endpoints:
- `POST /population/generate`: Generates synthetic population.
- `POST /network/build`: Builds and serializes social graph with nodes, edges, and degree telemetry.

---

## Configuration (`backend/config.py`)

Configured via environment variables with safe defaults:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint |
| `MODEL` | `qwen2.5vl:7b` | Active LLM model |
| `USE_LLM` | `true` | Toggle for LLM inference vs rule fallback |
| `N_PEOPLE` | `120` | Default synthetic audience population size |
| `SEED` | `42` | Random seed for deterministic reproducibility |
| `MAX_WAVES` | `6` | Maximum diffusion cascade horizon |

---

## Getting Started

### 1. Installation

```bash
python3 -m pip install -r requirements.txt
```

### 2. Run Tests

```bash
python3 -m pytest -v
```

### 3. Start the Server

```bash
uvicorn backend.main:app --reload --port 8000
```

- **Dashboard UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Health & Ollama Telemetry**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Precomputed Cache API**: [http://127.0.0.1:8000/demo/precomputed](http://127.0.0.1:8000/demo/precomputed)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Architecture & System Pipeline

Persona connects six modular computational stages:

```
[Candidate Message]
        │
        ▼
[1. Safety & Sanitization (backend/safety.py)]
        │  • Max 500 chars, control character stripping
        │  • XML data isolation (<message>...</message>) & jailbreak guard
        │  • Prohibits rewriting or optimizing harmful packs
        ▼
[2. Feature Extraction (backend/analyzer.py)]
        │  • 60% Local Ollama LLM (Qwen2.5-VL:7B) + 40% Deterministic Rule Engine
        │  • Urgency, authority claim, emotional pull, CTA strength, red flags, legit cues
        ▼
[3. Audience & Network Generation (backend/population.py, network.py)]
        │  • Archetype distributions with Gaussian trait jitter (SD = 0.08)
        │  • NetworkX social topology: homophily, family cliques (3-5 nodes), 5-6 community hubs
        │  • Channel transmission physics: WhatsApp, SMS, Social Feed, Call
        ▼
[4. Hybrid Reaction Modeling (backend/reactions.py)]
        │  • Behavioral bounds: act <= engage <= notice
        │  • Archetype base logit blend (0.6 LLM + 0.4 domain rules) + Person individual jitter
        ▼
[5. Multi-Wave Graph Cascade (backend/engine.py)]
        │  • Probabilistic discrete-time state sampling per wave
        │  • Forwarding decay (0.85^wave) x edge affinity (family 1.0, neighbor 0.6, hub 0.8)
        │  • Community immunity: reporting agents apply -0.05 belief friction to neighbors
        │  • Defensive intervention attenuation based on verification & clarity
        ▼
[6. Scoring, Insights & UI Telemetry (backend/scoring.py, insights.py, frontend/index.html)]
        • Basic reproduction number (R0), reach, act rate, time-to-peak
        • Demographic heatmap by cohort, age band, area, and language
        • Defensive recommendations matrix keyed by trigger x vulnerable group
        • Real-time SSE stage stream & interactive D3.js v7 force-directed network
```

### Dual Runtime Modes

1. **Demo Mode (Offline Cache)**:
   - Uses precomputed simulation bundles (`data/cache/precomputed.json` and `frontend/precomputed.json`) covering all 5 domain packs $\times$ 3 sample messages $\times$ 3 audience presets (45 combinations) plus 3 representative Monte Carlo intervention comparisons.
   - Guaranteed response time in **$<100$ milliseconds** ($<5$s requirement).
   - Functions 100% offline with zero external network connectivity or LLM daemon requirements.

2. **Live Mode (Ollama LLM)**:
   - Engages local Ollama model (`qwen2.5vl:7b`) with temperature-controlled JSON schema constraints.
   - Streams live Server-Sent Events (`GET /stream/{job_id}`) across all 6 pipeline stages.
   - Built-in 60s timeout: automatically falls back to rule-only mode with `degraded=true` if Ollama times out or encounters memory pressure.

---

## How Scores Are Computed

Persona evaluates information diffusion through quantitative epidemiological and behavioral metrics:

### 1. Basic Reproduction Number ($R_0$)
Measures the average number of secondary exposures generated per sharing node:
$$R_0 = \frac{\sum_{w=1}^{W} \text{NewExposures}_w}{\max\left(1, \; \sum_{w=0}^{W-1} \text{UniqueSharers}_w\right)}$$
- $R_0 < 1.0$: Subcritical diffusion; message dies out and remains contained.
- $R_0 \ge 1.0$: Supercritical diffusion; self-sustaining outbreak across community clusters.

### 2. Reach Fraction
The cumulative proportion of the population exposed to the message:
$$\text{Reach Fraction} = \frac{|\bigcup_{w=0}^{W} \text{ReachedNodes}_w|}{N_{\text{population}}}$$

### 3. Act Fraction (Victim / Conversion Rate)
The proportion of reached agents who took the primary domain action (e.g., clicked phishing link, paid scam fee, or attended clinic):
$$\text{Act Fraction} = \frac{\sum \text{Actors}}{\max\left(1, \; |\bigcup_{w=0}^{W} \text{ReachedNodes}_w|\right)}$$

### 4. Time-to-Peak
The discrete wave index $w^*$ that experienced the maximal inflow of newly exposed individuals:
$$w^* = \arg\max_w \left( |\text{NewlyReached}_w| \right)$$

### 5. Verdict Tier Thresholds
Verdicts map cascade metrics ($R_0$ and Reach) to domain pack tiers:
- **Tier 0 (Contained / Ignored)**: $\text{Reach} < 0.10$ and $R_0 < 0.70$
- **Tier 1 (Niche Hit / Niche Reach)**: $\text{Reach} \ge 0.10$ or $R_0 \ge 0.70$
- **Tier 2 (Wide Spread / Strong Reach)**: $\text{Reach} \ge 0.30$ or $R_0 \ge 1.00$
- **Tier 3 (Outbreak / Breakout)**: $\text{Reach} \ge 0.55$ or $R_0 \ge 1.40$

---

## Key Modeling Assumptions

1. **Cognitive Hierarchy**: Behavioral responses obey strict human cognitive bounds: an agent cannot act without engaging, and cannot engage without noticing ($\text{act} \le \text{engage} \le \text{notice}$).
2. **Channel Structure**: Information cannot diffuse through non-existent social ties. WhatsApp creates tight small-world clusters ($C > 0.15$); SMS creates sparse links; Social Feeds operate via scale-free hub broadcasts; phone calls have zero forwarding.
3. **Transmission Decay**: Information novelty degrades over time: forwarding probability attenuates as $0.85^{\text{wave}}$.
4. **Relational Affinity**: Transmission probabilities depend strongly on interpersonal trust: family edges transmit with factor $1.0$, neighborhood ties with $0.6$, and community hub broadcasts with $0.8$.
5. **Community Immunity**: Vigilant individuals who report suspicious messages introduce cognitive friction among their immediate social circle, lowering neighbors' susceptibility by $0.05$ in subsequent waves.

---

## Limitations

- **Synthetic Behavior vs. Empirical Ground Truth**: Agents are simulated computational entities driven by demographic trait distributions and LLM psychological reasoning. They do not possess real autobiographical memories, financial histories, or personal relationships.
- **Cross-Platform Bleed**: Real-world rumors often cross multiple platforms simultaneously (e.g. YouTube video $\to$ WhatsApp forward $\to$ Twitter discussion). Persona models single-channel cascades per run.
- **Closed Topology**: Simulations run on bounded community networks ($N=120$ to $N=500$) rather than open, infinite web graphs.

---

## Honest Scenario Notice: Projections, Not Forecasts

> [!IMPORTANT]
> **Persona simulation outputs are synthetic comparative scenarios, NOT absolute real-world forecasts or deterministic predictions.**
>
> When comparing two messages or two interventions (e.g. Candidate A vs Candidate B), the winner, percentage difference, and $R_0$ metrics reflect **directional behavioral friction under identical structural assumptions**. They help researchers and policy designers identify vulnerabilities and optimize messaging clarity; they do not predict exact future victim counts or conversion rates in real-world deployments.

---

## Responsible Use Policy

Persona is an analytical research and defense tool designed to evaluate communication vulnerabilities and test defensive counter-messaging.

1. **Strict Guardrails**: Persona strictly refuses and prevents any requests to generate, optimize, rewrite, or polish fraudulent, misleading, or deceptive content (`backend/safety.py`).
2. **Defensive Guidance Only**: For harmful packs (`scam_fraud`, `misinformation`), the system exclusively produces defensive friction recommendations (e.g. call-back verification steps, automated bank cooldowns, vernacular advisories). Message improvement tips are strictly restricted to beneficial public interest domains (`health_campaign`, `civic_policy`).
3. **Data Isolation**: All candidate message text is treated strictly as passive data inside XML delimiters (`<message>...</message>`) to neutralize prompt injection or jailbreak attempts.

