"""Main entrypoint for Persona FastAPI server.

Exposes:
- POST /simulate: Run end-to-end message spread simulation as a background job.
- GET /stream/{job_id}: Server-Sent Events (SSE) streaming pipeline progress stages.
- POST /compare: Compare two interventions or messages across N=20 Monte Carlo runs.
- GET /packs: List available domain packs without internal weights.
- GET /presets: List curated audience presets.
- GET /health: Health check and Ollama connectivity verification.
- Static mount at "/": Serves frontend HTML dashboard.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests

from backend.analyzer import analyze_message, sanitize_input
from backend.archetypes import PRESETS, generate_archetypes
from backend.config import (
    CACHE_DIR,
    DEFAULT_R0_THRESHOLDS,
    DEFAULT_REACH_THRESHOLDS,
    FRONTEND_DIR,
    MAX_WAVES,
    MODEL,
    N_PEOPLE,
    OLLAMA_URL,
    SEED,
    USE_LLM,
)
from backend.domain_packs import PACKS, SCAM_FRAUD, list_public_packs
from backend.engine import run_simulation
from backend.insights import compare_interventions, generate_strategic_insights
from backend.models import Archetype, MessageFeatures, Person, Reaction, SimulationResult
from backend.network import build_network
from backend.population import generate_population
from backend.reactions import get_reactions
from backend.safety import RESPONSIBLE_USE_NOTE
from backend.scoring import score_cascade

logger = logging.getLogger("persona.main")
logging.basicConfig(level=logging.INFO)

# Load precomputed bundle for offline demo mode
PRECOMPUTED_FILE = CACHE_DIR / "precomputed.json"
PRECOMPUTED_BUNDLE: Dict[str, Any] = {}
if PRECOMPUTED_FILE.exists():
    try:
        with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
            PRECOMPUTED_BUNDLE = json.load(f)
        logger.info(f"Loaded precomputed bundle with {len(PRECOMPUTED_BUNDLE.get('simulations', {}))} simulation cascades.")
    except Exception as exc:
        logger.warning(f"Could not load precomputed bundle: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager initiating background Ollama pre-warm probe."""
    asyncio.create_task(prewarm_ollama())
    yield


app = FastAPI(
    title="Persona API",
    description="Synthetic Audience Message Spread Simulator",
    version="0.2.0",
    lifespan=lifespan,
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Rate Limiting & Background Job State Store
# ---------------------------------------------------------------------------

class JobState:
    """In-memory state and event queue for background simulation jobs."""

    def __init__(self, job_id: str, req: SimulateRequest):
        self.job_id = job_id
        self.req = req
        self.status: str = "running"
        self.created_at: float = time.time()
        self.degraded: bool = False
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.event_history: List[Dict[str, Any]] = []

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an event to history and push to queue for streaming listeners."""
        event_obj = {"event_type": event_type, "data": data, "timestamp": time.time()}
        self.event_history.append(event_obj)
        self.queue.put_nowait(event_obj)


# Global concurrency control: rate limit to 1 active concurrent simulation/comparison job
concurrency_lock = asyncio.Lock()
active_job_id: Optional[str] = None
jobs: Dict[str, JobState] = {}


# ---------------------------------------------------------------------------
# Startup Pre-warm Handler
# ---------------------------------------------------------------------------

async def prewarm_ollama() -> None:
    """Pre-warm local Ollama model on startup with a single lightweight token call."""
    logger.info(f"Initiating Ollama pre-warm probe for model '{MODEL}' at {OLLAMA_URL}...")
    try:
        def _ping():
            return requests.post(
                f"{OLLAMA_URL.rstrip('/')}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=4.0,
            )

        resp = await asyncio.to_thread(_ping)
        if resp.status_code == 200:
            logger.info(f"Ollama pre-warm successful. Model '{MODEL}' loaded into memory.")
        else:
            logger.warning(f"Ollama pre-warm returned HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        logger.warning(f"Ollama pre-warm skipped or offline: {exc}")


# ---------------------------------------------------------------------------
# Health, Packs, and Presets
# ---------------------------------------------------------------------------

@app.get("/health")
def get_health():
    """Health check endpoint that verifies server liveness, concurrency, and pings Ollama."""
    ollama_info = {
        "url": OLLAMA_URL,
        "configured_model": MODEL,
        "connected": False,
        "latency_ms": None,
        "available_models": [],
        "error": None,
    }

    start = time.perf_counter()
    try:
        response = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3.0)
        latency = round((time.perf_counter() - start) * 1000, 2)
        if response.status_code == 200:
            data = response.json()
            models_list = [m.get("name") for m in data.get("models", [])]
            ollama_info["connected"] = True
            ollama_info["latency_ms"] = latency
            ollama_info["available_models"] = models_list
        else:
            ollama_info["error"] = f"HTTP {response.status_code}: {response.text[:120]}"
    except Exception as exc:
        ollama_info["error"] = str(exc)

    return {
        "status": "ok",
        "app": "persona",
        "active_job": active_job_id,
        "ollama": ollama_info,
        "config": {
            "n_people": N_PEOPLE,
            "seed": SEED,
            "max_waves": MAX_WAVES,
            "use_llm": USE_LLM,
            "demo_cache_available": bool(PRECOMPUTED_BUNDLE),
        },
        "responsible_use_note": RESPONSIBLE_USE_NOTE,
    }


@app.get("/packs")
def get_packs():
    """Return domain packs without internal weights."""
    return list_public_packs()


@app.get("/presets")
def get_presets():
    """Return all hand-curated audience presets and their archetypes."""
    return {pid: [a.model_dump() for a in archs] for pid, archs in PRESETS.items()}


@app.get("/demo/precomputed")
def get_precomputed_data():
    """Return complete precomputed cache bundle for instant offline Demo Mode."""
    if not PRECOMPUTED_BUNDLE and PRECOMPUTED_FILE.exists():
        with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return PRECOMPUTED_BUNDLE


# ---------------------------------------------------------------------------
# Simulation Pipeline Models & Execution
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    message: str = Field(..., description="Message text to simulate (max 500 characters)")
    pack_id: str = Field(default="scam_fraud", description="Domain pack ID")
    audience_preset: Optional[str] = Field(default=None, description="Preset ID (e.g. urban_seniors)")
    audience_description: Optional[str] = Field(default=None, description="Custom audience description")
    channel: Optional[str] = Field(default=None, description="Communication channel (whatsapp, sms, social_feed, call)")
    intervention: Optional[str] = Field(default=None, description="Optional warning or defensive clarification text")
    use_llm: Optional[bool] = Field(default=None, description="Toggle LLM reasoning vs deterministic rule engine")
    demo_mode: bool = Field(default=False, description="Use offline precomputed cache for instant response")
    seed: int = Field(default=SEED, description="Simulation seed")
    n_people: int = Field(default=N_PEOPLE, ge=10, le=500, description="Synthetic population size")


def execute_pipeline(
    job: JobState,
    req: SimulateRequest,
    use_llm: bool,
) -> Dict[str, Any]:
    """Execute end-to-end simulation stages synchronously and report progress events."""
    pack = PACKS.get(req.pack_id, SCAM_FRAUD)

    # Stage 1: Reading message
    job.record_event("progress", {
        "stage": "Reading message",
        "progress": 0.10,
        "detail": "Extracting message features, claims, and suspicious cues",
    })
    features = analyze_message(req.message, pack_id=pack.id, use_llm=use_llm)

    # Stage 2: Designing audience
    job.record_event("progress", {
        "stage": "Designing audience",
        "progress": 0.25,
        "detail": "Configuring target audience demographic segments",
    })
    if req.audience_description and req.audience_description.strip():
        if use_llm:
            try:
                archetypes = generate_archetypes(req.audience_description.strip(), n=10)
            except Exception as e:
                logger.warning(f"Audience generation failed ({e}), falling back to preset.")
                preset_key = req.audience_preset or "urban_seniors"
                archetypes = PRESETS.get(preset_key, PRESETS["urban_seniors"])
        else:
            preset_key = req.audience_preset or "urban_seniors"
            archetypes = PRESETS.get(preset_key, PRESETS["urban_seniors"])
    else:
        preset_key = req.audience_preset or "urban_seniors"
        archetypes = PRESETS.get(preset_key, PRESETS["urban_seniors"])

    # Stage 3: Segments reacting (i/12)
    total_segs = len(archetypes)
    for i, arch in enumerate(archetypes, 1):
        job.record_event("progress", {
            "stage": f"Segments reacting ({i}/{total_segs})",
            "progress": round(0.25 + 0.35 * (i / max(1, total_segs)), 2),
            "segment": arch.label,
        })

    reactions = get_reactions(
        features=features,
        message=req.message,
        archetypes=archetypes,
        pack=pack,
        k=1 if not use_llm else 3,
        use_cache=True,
        use_llm=use_llm,
    )

    # Stage 4: Building network
    channel = req.channel or features.channel_guess or "whatsapp"
    job.record_event("progress", {
        "stage": "Building network",
        "progress": 0.65,
        "detail": f"Synthesizing social graph topology for {channel}",
        "channel": channel,
    })
    people = generate_population(archetypes, n=req.n_people, seed=req.seed)
    net_res = build_network(people, channel=channel, seed=req.seed)

    # Stage 5: Spreading wave n
    def wave_cb(wave_idx: int):
        job.record_event("progress", {
            "stage": f"Spreading wave {wave_idx}",
            "progress": round(0.70 + 0.04 * wave_idx, 2),
            "wave": wave_idx,
        })

    sim_result = run_simulation(
        people=people,
        graph=net_res.graph,
        reactions=reactions,
        features=features,
        pack=pack,
        intervention=req.intervention,
        seed=req.seed,
        wave_callback=wave_cb,
    )

    # Stage 6: Scoring
    job.record_event("progress", {
        "stage": "Scoring",
        "progress": 0.95,
        "detail": "Calculating R0, demographic vulnerability, and strategic defense",
    })
    scoring_data = score_cascade(
        people=people,
        waves=sim_result.waves,
        edges=sim_result.edges,
        reactions=reactions,
        pack=pack,
        node_states=sim_result.node_states,
    )

    insights_data = generate_strategic_insights(
        simulation_result=sim_result,
        pack=pack,
        strongest_trigger=scoring_data["strongest_trigger"],
        weakest_group=scoring_data["weakest_group"],
        features=features,
    )

    final_result = {
        "job_id": job.job_id,
        "status": "completed",
        "degraded": False,
        "message": req.message,
        "pack_id": pack.id,
        "channel": channel,
        "features": features.model_dump(),
        "audience": [a.model_dump() for a in archetypes],
        "population_count": len(people),
        "network_summary": {
            "node_count": len(people),
            "edge_count": len(net_res.graph.edges),
            "channel": channel,
        },
        "network_payload": net_res.payload,
        "reactions": {aid: r.model_dump() for aid, r in reactions.items()},
        "simulation": sim_result.model_dump(),
        "scoring": scoring_data,
        "insights": insights_data,
    }

    return final_result


async def run_simulation_pipeline_task(job_id: str, req: SimulateRequest) -> None:
    """Asynchronous background coordinator enforcing global 60s timeout and degraded fallback."""
    global active_job_id
    job = jobs[job_id]
    use_llm = req.use_llm if req.use_llm is not None else USE_LLM
    degraded = False

    try:
        if use_llm:
            try:
                # Enforce global 60s timeout on LLM pipeline
                final_result = await asyncio.wait_for(
                    asyncio.to_thread(execute_pipeline, job, req, True),
                    timeout=60.0,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning(f"Simulation {job_id} LLM execution failed/timed out ({exc}). Falling back to rule-only mode.")
                degraded = True
                job.record_event("progress", {
                    "stage": "Degraded Fallback",
                    "progress": 0.50,
                    "detail": "LLM timed out after 60s or failed; completing simulation with deterministic rule engine.",
                    "degraded": True,
                })
                # Graceful fallback to rule-only mode
                final_result = await asyncio.to_thread(execute_pipeline, job, req, False)
                final_result["degraded"] = True
        else:
            final_result = await asyncio.to_thread(execute_pipeline, job, req, False)

        final_result["degraded"] = degraded or final_result.get("degraded", False)
        job.status = "completed"
        job.degraded = final_result["degraded"]
        job.result = final_result

        # Final event containing full result JSON
        job.record_event("complete", final_result)

    except Exception as fatal_exc:
        logger.error(f"Fatal unrecoverable error in simulation {job_id}: {fatal_exc}", exc_info=True)
        job.status = "failed"
        job.error = str(fatal_exc)
        job.record_event("error", {"error": str(fatal_exc), "status": "failed"})

    finally:
        # Clear concurrency lock
        active_job_id = None


@app.post("/simulate")
async def api_simulate(req: SimulateRequest, background_tasks: BackgroundTasks):
    """Initiate message spread simulation pipeline as an asynchronous background task.

    Returns job_id to stream real-time progress events via GET /stream/{job_id}.
    """
    global active_job_id

    # 1. Rate Limit: 1 concurrent job
    if active_job_id is not None:
        raise HTTPException(
            status_code=429,
            detail="A simulation job is currently in progress. Only 1 concurrent job is allowed.",
        )

    # 2. Request Validation: 500 char limit & empty checks
    msg_clean = req.message.strip() if req.message else ""
    if not msg_clean:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(msg_clean) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Message exceeds maximum allowed length of 500 characters (length={len(msg_clean)}).",
        )

    if req.intervention and len(req.intervention.strip()) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Intervention exceeds maximum allowed length of 500 characters (length={len(req.intervention.strip())}).",
        )

    if req.audience_description and len(req.audience_description.strip()) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Audience description exceeds maximum allowed length of 500 characters (length={len(req.audience_description.strip())}).",
        )

    if req.pack_id not in PACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pack '{req.pack_id}'. Valid packs are: {list(PACKS.keys())}",
        )

    # 2.5 Fast Path: Offline Demo Mode
    if req.demo_mode and PRECOMPUTED_BUNDLE:
        sims = PRECOMPUTED_BUNDLE.get("simulations", {})
        target_preset = req.audience_preset or "urban_seniors"
        match = None
        for k, v in sims.items():
            if v.get("pack_id") == req.pack_id and v.get("audience_preset") == target_preset:
                if v.get("message", "").strip() == msg_clean:
                    match = v
                    break
        if not match:
            for k, v in sims.items():
                if v.get("pack_id") == req.pack_id and v.get("audience_preset") == target_preset:
                    match = v
                    break
        if not match:
            for k, v in sims.items():
                if v.get("pack_id") == req.pack_id:
                    match = v
                    break
        if not match and sims:
            match = next(iter(sims.values()))

        if match:
            demo_job_id = f"demo_{uuid.uuid4().hex[:8]}"
            res_copy = dict(match)
            res_copy["job_id"] = demo_job_id
            job = JobState(job_id=demo_job_id, req=req)
            job.status = "completed"
            job.result = res_copy
            jobs[demo_job_id] = job
            job.record_event("complete", res_copy)
            return {
                "job_id": demo_job_id,
                "status": "completed",
                "stream_url": f"/stream/{demo_job_id}",
                "result": res_copy,
                "cached": True,
                "message": "Demo mode: loaded from offline precomputed cache.",
            }

    # 3. Create Job and Spawn Task
    job_id = f"sim_{uuid.uuid4().hex[:10]}"
    job = JobState(job_id=job_id, req=req)
    jobs[job_id] = job
    active_job_id = job_id

    background_tasks.add_task(run_simulation_pipeline_task, job_id, req)

    return {
        "job_id": job_id,
        "status": "running",
        "stream_url": f"/stream/{job_id}",
        "message": "Simulation started as background task.",
    }


@app.get("/stream/{job_id}")
async def stream_simulation_events(job_id: str):
    """Stream real-time Server-Sent Events (SSE) for pipeline execution stages.

    Stages:
    - "Reading message"
    - "Designing audience"
    - "Segments reacting (i/12)"
    - "Building network"
    - "Spreading wave n"
    - "Scoring"
    Final event contains the full result JSON.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Simulation job '{job_id}' not found.")

    async def event_generator():
        sent_count = 0

        # Yield previously recorded events first to handle client connection delays
        for ev in job.event_history:
            event_type = ev.get("event_type", "progress")
            yield f"event: {event_type}\ndata: {json.dumps(ev['data'])}\n\n"
            sent_count += 1
            if event_type in ("complete", "error"):
                return

        # Listen to live event queue
        while True:
            if job.status == "completed" and job.result is not None:
                # If complete event wasn't sent yet
                if not any(e.get("event_type") == "complete" for e in job.event_history):
                    yield f"event: complete\ndata: {json.dumps(job.result)}\n\n"
                return
            elif job.status == "failed":
                yield f"event: error\ndata: {json.dumps({'error': job.error, 'status': 'failed'})}\n\n"
                return

            try:
                ev = await asyncio.wait_for(job.queue.get(), timeout=2.0)
                event_type = ev.get("event_type", "progress")
                yield f"event: {event_type}\ndata: {json.dumps(ev['data'])}\n\n"
                if event_type in ("complete", "error"):
                    return
            except asyncio.TimeoutError:
                # Keep-alive heartbeat ping
                if job.status in ("completed", "failed"):
                    return
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    """Retrieve current status or final output of a simulation job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Simulation job '{job_id}' not found.")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "degraded": job.degraded,
        "error": job.error,
        "result": job.result,
    }


# ---------------------------------------------------------------------------
# Compare Endpoint (Monte Carlo N=20 runs)
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    message: str = Field(..., description="Message text to evaluate (max 500 characters)")
    pack_id: str = Field(default="scam_fraud", description="Domain pack ID")
    audience: Optional[str] = Field(default="urban_seniors", description="Preset ID or audience description")
    interventions: List[str] = Field(..., description="List of interventions to compare (minimum 2)")
    n_runs: int = Field(default=20, ge=1, le=50, description="Monte Carlo run count")
    n_people: int = Field(default=60, ge=10, le=200)
    channel: str = Field(default="whatsapp")
    use_llm: Optional[bool] = Field(default=None)
    demo_mode: bool = Field(default=False, description="Use offline precomputed demo cache")


@app.post("/compare")
async def api_compare(req: CompareRequest):
    """Run comparative Monte Carlo simulations across two interventions (N=20 runs)."""
    global active_job_id

    # 1. Fast Path: Offline Demo Mode
    if req.demo_mode and PRECOMPUTED_BUNDLE:
        demos = PRECOMPUTED_BUNDLE.get("compare_demos", {})
        for spec_id, demo_res in demos.items():
            if demo_res.get("pack_id") == req.pack_id:
                return demo_res
        if demos:
            return next(iter(demos.values()))

    # 2. Rate Limit
    if active_job_id is not None:
        raise HTTPException(
            status_code=429,
            detail="A simulation job is currently in progress. Only 1 concurrent job is allowed.",
        )

    # 3. Validation
    msg_clean = req.message.strip() if req.message else ""
    if not msg_clean:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(msg_clean) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Message exceeds maximum allowed length of 500 characters (length={len(msg_clean)}).",
        )

    if not req.interventions or len(req.interventions) < 2:
        raise HTTPException(
            status_code=400,
            detail="Must provide at least two interventions to compare (e.g. ['Intervention A', 'Intervention B']).",
        )

    for i, inv in enumerate(req.interventions[:2]):
        if len(inv.strip()) > 500:
            raise HTTPException(
                status_code=400,
                detail=f"Intervention {i+1} exceeds maximum allowed length of 500 characters.",
            )

    if req.pack_id not in PACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pack '{req.pack_id}'. Valid packs are: {list(PACKS.keys())}",
        )

    # 3. Resolve Audience
    aud_key = (req.audience or "urban_seniors").strip()
    if aud_key in PRESETS:
        archetypes = PRESETS[aud_key]
    else:
        try:
            archetypes = generate_archetypes(aud_key, n=10)
        except Exception:
            archetypes = PRESETS["urban_seniors"]

    # 4. Acquire Lock & Execute Monte Carlo
    pack = PACKS[req.pack_id]
    active_job_id = f"comp_{uuid.uuid4().hex[:8]}"
    use_llm = req.use_llm if req.use_llm is not None else USE_LLM

    try:
        comparison_res = await asyncio.to_thread(
            compare_interventions,
            message=msg_clean,
            pack=pack,
            archetypes=archetypes,
            interventions=req.interventions[:2],
            n_runs=req.n_runs,
            n_people=req.n_people,
            channel=req.channel,
            use_llm=use_llm,
        )
        return comparison_res
    finally:
        active_job_id = None


# ---------------------------------------------------------------------------
# Auxiliary Endpoints (Audience, Network, Population, Analysis)
# ---------------------------------------------------------------------------

class CustomAudienceRequest(BaseModel):
    audience_description: str
    n: int = Field(default=10, ge=4, le=20)


@app.post("/archetypes/generate")
def api_generate_archetypes(req: CustomAudienceRequest):
    """Generate custom archetypes from user description via Ollama LLM."""
    archetypes = generate_archetypes(req.audience_description, n=req.n)
    return {"archetypes": [a.model_dump() for a in archetypes]}


class GeneratePopulationRequest(BaseModel):
    preset_id: Optional[str] = None
    archetypes: Optional[List[Archetype]] = None
    n: int = Field(default=N_PEOPLE, ge=10, le=500)
    seed: int = Field(default=SEED)


@app.post("/population/generate")
def api_generate_population(req: GeneratePopulationRequest):
    """Sample a synthetic population of Person models."""
    archs = req.archetypes
    if not archs:
        preset_key = req.preset_id or "urban_seniors"
        archs = PRESETS.get(preset_key, PRESETS["urban_seniors"])
    people = generate_population(archs, n=req.n, seed=req.seed)
    return {"people": [p.model_dump() for p in people], "count": len(people)}


class BuildNetworkRequest(BaseModel):
    people: List[Person]
    channel: str = Field(default="whatsapp")
    seed: int = Field(default=SEED)


@app.post("/network/build")
def api_build_network(req: BuildNetworkRequest):
    """Build a NetworkX social graph and return serializable nodes/edges/metrics."""
    res = build_network(req.people, channel=req.channel, seed=req.seed)
    return res.payload


class AnalyzeMessageRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze")
    pack_id: Optional[str] = Field(default=None, description="Optional domain pack ID")


@app.post("/analyze")
def api_analyze_message(req: AnalyzeMessageRequest):
    """Analyze a candidate message for deceptive cues, tone, authority, and behavioral features."""
    features = analyze_message(req.text, pack_id=req.pack_id)
    return {"features": features.model_dump()}


class SimulateReactionsRequest(BaseModel):
    message: str = Field(..., description="Candidate message text")
    features: Optional[MessageFeatures] = Field(default=None, description="Precomputed features")
    pack_id: str = Field(default="scam_fraud", description="Target domain pack ID")
    preset_id: Optional[str] = Field(default=None, description="Audience preset ID")
    archetypes: Optional[List[Archetype]] = Field(default=None, description="Custom archetypes")
    k: int = Field(default=1, ge=1, le=5, description="Number of LLM samples to average")


@app.post("/reactions/simulate")
def api_simulate_reactions(req: SimulateReactionsRequest):
    """Compute blended archetype reactions for a given message and audience."""
    pack = PACKS.get(req.pack_id, SCAM_FRAUD)
    features = req.features or analyze_message(req.message, pack_id=req.pack_id)
    archetypes = req.archetypes
    if not archetypes:
        preset_key = req.preset_id or "urban_seniors"
        archetypes = PRESETS.get(preset_key, PRESETS["urban_seniors"])

    reactions = get_reactions(features, req.message, archetypes, pack, k=req.k)
    return {
        "pack_id": pack.id,
        "reactions": {aid: r.model_dump() for aid, r in reactions.items()},
    }


# ---------------------------------------------------------------------------
# Static Web Dashboard Mount
# ---------------------------------------------------------------------------

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
