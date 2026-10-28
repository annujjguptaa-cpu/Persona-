"""
domain_packs.py  -  Persona
===========================
A "domain pack" tells the simulator what a message MEANS in a given context:
  * what "acting" is (pays, vaccinates, buys ...)
  * whether wide spread is GOOD (beneficial) or BAD (harmful)
  * how personality traits and message features change each reaction
  * what the verdict labels are called
  * sample messages for the demo (always including one legitimate message)

All numbers are starting points. Tune them in tests/calibration.py so that
rankings make sense (scam > legit for vulnerable people, etc.).
The sample messages are SIMULATION INPUTS only; Persona never generates or
improves scam or misinformation text.
"""
from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Universal vocabulary
# --------------------------------------------------------------------------
TRAITS: List[str] = [
    "attention",            # how much they notice messages
    "digital_literacy",     # comfort with phones, links, apps
    "trust_in_source",      # default trust in official-sounding senders
    "skepticism",           # tendency to doubt / verify
    "emotional_reactivity", # how strongly emotion drives them
    "social_sharing",       # habit of forwarding to family / groups
    "time_pressure",        # busy, rushed, acts without checking
    "domain_interest",      # interest in the topic of the message
    "financial_cushion",    # able to absorb a small loss / cost
    "language_barrier",     # mismatch with the message's language
]

ACTIONS: List[str] = ["engage", "act", "share", "report"]

FEATURES: List[str] = [
    "urgency",
    "emotional_pull",
    "authority_claim",   # 0 = none, 1 = strong official claim
    "call_to_action",
    "has_link",          # 0 or 1
]

# Logistic scaling constants (tune in tests/calibration.py).
#   logit(p) = logit(base) + TRAIT_K * sum(w_t * (trait - 0.5)) + FEATURE_K * sum(w_f * feature)
# Working in logit space keeps p inside 0-1 and lets traits and message
# strength combine naturally (a weak message stays weak even for vulnerable people).
TRAIT_K: float = 4.0
FEATURE_K: float = 2.2

Valence = Literal["harmful", "beneficial"]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class AudiencePreset(BaseModel):
    id: str            # must match a key in archetypes.PRESETS
    label: str
    description: str


class Sample(BaseModel):
    label: str
    kind: Literal["typical", "legit"]
    text: str = Field(max_length=500)


class TierThresholds(BaseModel):
    """Three cut-points -> four tiers. Tier = max(tier_by_reach, tier_by_r0)."""
    reach: List[float]   # fraction of audience exposed, ascending
    r0: List[float]      # ascending

    @model_validator(mode="after")
    def _check(self):
        for name in ("reach", "r0"):
            v = getattr(self, name)
            if len(v) != 3 or v != sorted(v):
                raise ValueError(f"{name} must be 3 ascending numbers")
        return self


class DomainPack(BaseModel):
    id: str
    title: str
    description: str
    act_label: str                          # what "act" means here
    valence: Valence                        # is wide spread good or bad?
    allow_message_improvement: bool         # False for harmful packs
    base_rates: Dict[str, float]            # starting probabilities per action
    trait_weights: Dict[str, Dict[str, float]]    # action -> trait -> weight
    feature_weights: Dict[str, Dict[str, float]]  # action -> feature -> weight
    tier_thresholds: TierThresholds
    verdict_labels: List[str]               # exactly 4, low -> high spread
    audience_presets: List[AudiencePreset]  # exactly 3
    samples: List[Sample]                   # exactly 3, at least one legit
    llm_context: str                        # one line injected into LLM prompts

    @model_validator(mode="after")
    def _validate(self):
        assert len(self.verdict_labels) == 4, f"{self.id}: need 4 verdict labels"
        assert len(self.audience_presets) == 3, f"{self.id}: need 3 presets"
        assert len(self.samples) == 3, f"{self.id}: need 3 samples"
        assert any(s.kind == "legit" for s in self.samples), f"{self.id}: need a legit sample"
        assert set(self.base_rates) == set(ACTIONS), f"{self.id}: base_rates keys"
        for a, tw in self.trait_weights.items():
            assert a in ACTIONS, f"{self.id}: unknown action {a}"
            assert set(tw) <= set(TRAITS), f"{self.id}: unknown trait in {a}"
        for a, fw in self.feature_weights.items():
            assert a in ACTIONS, f"{self.id}: unknown action {a}"
            assert set(fw) <= set(FEATURES), f"{self.id}: unknown feature in {a}"
        return self


# --------------------------------------------------------------------------
# Shared audience presets (ids map to archetypes.PRESETS)
# --------------------------------------------------------------------------
P_SENIORS = AudiencePreset(
    id="urban_seniors", label="Urban seniors",
    description="Retired people aged 60+ in cities; often caring for a spouse or relative; trust official-sounding senders.")
P_STUDENTS = AudiencePreset(
    id="college_students", label="College students",
    description="Ages 18-24, very online, heavy group-chat users, short on money, quick to share.")
P_FARMERS = AudiencePreset(
    id="rural_farmers", label="Rural farmers",
    description="Small and mid-size farmers in villages; basic phones or first smartphones; regional language preferred.")
P_PARENTS = AudiencePreset(
    id="working_parents", label="Working parents",
    description="Ages 28-45, urban, time-poor, juggling kids and jobs, active in family and school groups.")
P_SHOPKEEPERS = AudiencePreset(
    id="small_shopkeepers", label="Small shopkeepers",
    description="Local traders using UPI daily; community hubs; sensitive to payments and offers.")
P_HEALTHWORKERS = AudiencePreset(
    id="community_health_workers", label="Community health workers",
    description="ASHA and anganwadi workers; trusted local voices; relay information to many households.")
P_GIGWORKERS = AudiencePreset(
    id="gig_workers", label="Gig and daily-wage workers",
    description="Delivery riders, drivers, daily-wage workers; phone-first, low cushion, long hours.")
P_PROFESSIONALS = AudiencePreset(
    id="urban_professionals", label="Urban professionals",
    description="Ages 25-40, tech-comfortable, sceptical of obvious spam, time-pressed, active on social feeds.")
P_HOMEMAKERS = AudiencePreset(
    id="homemakers", label="Homemakers",
    description="Household decision-makers active in family and neighbourhood groups; moderate tech comfort.")

# --------------------------------------------------------------------------
# PACK 1: scam and fraud (harmful)
# --------------------------------------------------------------------------
SCAM_FRAUD = DomainPack(
    id="scam_fraud",
    title="Scam and fraud",
    description="How a suspicious payment or verification message spreads and who falls for it. Defensive analysis only.",
    act_label="pays or clicks",
    valence="harmful",
    allow_message_improvement=False,
    base_rates={"engage": 0.20, "act": 0.04, "share": 0.08, "report": 0.03},
    trait_weights={
        "engage": {"attention": 0.25, "emotional_reactivity": 0.25, "trust_in_source": 0.20,
                   "skepticism": -0.30, "time_pressure": 0.15},
        "act":    {"digital_literacy": -0.20, "trust_in_source": 0.25, "skepticism": -0.40,
                   "time_pressure": 0.25, "emotional_reactivity": 0.20, "financial_cushion": 0.05},
        "share":  {"social_sharing": 0.35, "emotional_reactivity": 0.20, "skepticism": -0.20,
                   "trust_in_source": 0.10},
        "report": {"skepticism": 0.30, "digital_literacy": 0.20, "trust_in_source": -0.10},
    },
    feature_weights={
        "engage": {"urgency": 0.20, "authority_claim": 0.20, "emotional_pull": 0.20, "has_link": 0.05},
        "act":    {"urgency": 0.25, "authority_claim": 0.20, "call_to_action": 0.25, "emotional_pull": 0.20, "has_link": 0.10},
        "share":  {"emotional_pull": 0.20, "urgency": 0.10},
        "report": {"has_link": 0.05},
    },
    tier_thresholds=TierThresholds(reach=[0.10, 0.30, 0.55], r0=[0.7, 1.0, 1.4]),
    verdict_labels=["Contained", "Niche hit", "Wide spread", "Outbreak"],
    audience_presets=[P_SENIORS, P_STUDENTS, P_GIGWORKERS],
    samples=[
        Sample(label="Fake claim fee (simulation sample)", kind="typical",
               text="Dear customer, your insurance claim is on hold. Pay Rs 499 processing fee today to release it. Pay now: http://claim-help.example"),
        Sample(label="Fake KYC update (simulation sample)", kind="typical",
               text="Your Ayushman card will be blocked in 24 hours. Complete KYC at http://ayush-kyc.example and share the OTP."),
        Sample(label="Genuine bank alert", kind="legit",
               text="Your account XX1234 was credited with Rs 2,500 on 24-Oct. Available balance Rs 18,320. Do not share OTP or PIN with anyone."),
    ],
    llm_context="Judge how likely each person is to be deceived by, forward, or report a possibly fraudulent message. Skeptics often verify or ignore. Be realistic, not alarmist.",
)

# --------------------------------------------------------------------------
# PACK 2: health campaign (beneficial)
# --------------------------------------------------------------------------
HEALTH_CAMPAIGN = DomainPack(
    id="health_campaign",
    title="Health campaign",
    description="How a public-health message (vaccination, screening, prevention) reaches people and moves them to act.",
    act_label="takes the health action (visits clinic, gets vaccinated, follows advice)",
    valence="beneficial",
    allow_message_improvement=True,
    base_rates={"engage": 0.25, "act": 0.08, "share": 0.10, "report": 0.01},
    trait_weights={
        "engage": {"attention": 0.25, "domain_interest": 0.30, "trust_in_source": 0.20,
                   "language_barrier": -0.25, "emotional_reactivity": 0.10},
        "act":    {"trust_in_source": 0.30, "domain_interest": 0.25, "language_barrier": -0.20,
                   "time_pressure": -0.15, "financial_cushion": 0.15, "skepticism": -0.10},
        "share":  {"social_sharing": 0.35, "trust_in_source": 0.15, "domain_interest": 0.15,
                   "emotional_reactivity": 0.10},
        "report": {"skepticism": 0.10},
    },
    feature_weights={
        "engage": {"emotional_pull": 0.15, "authority_claim": 0.15, "urgency": 0.05},
        "act":    {"call_to_action": 0.30, "authority_claim": 0.15, "urgency": 0.10, "has_link": -0.05},
        "share":  {"emotional_pull": 0.20, "authority_claim": 0.10},
        "report": {},
    },
    tier_thresholds=TierThresholds(reach=[0.15, 0.35, 0.60], r0=[0.7, 1.0, 1.3]),
    verdict_labels=["Ignored", "Niche reach", "Strong reach", "Breakout"],
    audience_presets=[P_PARENTS, P_FARMERS, P_HEALTHWORKERS],
    samples=[
        Sample(label="Free dengue test", kind="typical",
               text="Free dengue testing at the community health centre this Saturday, 9 AM to 1 PM. Bring your Aadhaar card. Please tell your neighbours."),
        Sample(label="Vaccination drive", kind="typical",
               text="Free measles vaccination for children under 5 at your anganwadi on Monday. Safe, quick, and 100% free."),
        Sample(label="Vague health notice", kind="legit",
               text="Health department: residents are advised to follow monsoon precautions and visit a nearby facility if unwell."),
    ],
    llm_context="Judge how likely each person is to notice, trust, act on, and share a public-health message. Consider language, trust in the sender, cost, and time.",
)

# --------------------------------------------------------------------------
# PACK 3: product and marketing (beneficial)
# --------------------------------------------------------------------------
PRODUCT_MARKETING = DomainPack(
    id="product_marketing",
    title="Product and marketing",
    description="How an app launch, offer, or announcement spreads and who converts. Useful for founders and small businesses.",
    act_label="buys or signs up",
    valence="beneficial",
    allow_message_improvement=True,
    base_rates={"engage": 0.18, "act": 0.04, "share": 0.07, "report": 0.01},
    trait_weights={
        "engage": {"attention": 0.25, "domain_interest": 0.35, "emotional_reactivity": 0.15,
                   "skepticism": -0.10},
        "act":    {"domain_interest": 0.35, "financial_cushion": 0.25, "digital_literacy": 0.15,
                   "skepticism": -0.20, "time_pressure": -0.10},
        "share":  {"social_sharing": 0.40, "domain_interest": 0.20, "emotional_reactivity": 0.15},
        "report": {"skepticism": 0.10},
    },
    feature_weights={
        "engage": {"emotional_pull": 0.20, "urgency": 0.10, "authority_claim": 0.05},
        "act":    {"call_to_action": 0.30, "urgency": 0.15, "has_link": 0.10},
        "share":  {"emotional_pull": 0.25},
        "report": {"urgency": 0.05},
    },
    tier_thresholds=TierThresholds(reach=[0.10, 0.30, 0.55], r0=[0.6, 1.0, 1.4]),
    verdict_labels=["Flop", "Niche hit", "Strong hit", "Viral"],
    audience_presets=[P_STUDENTS, P_PROFESSIONALS, P_SHOPKEEPERS],
    samples=[
        Sample(label="App launch", kind="typical",
               text="We just launched StudyBuddy, a free app that turns your notes into quizzes. First 500 students get premium for a month. Try it today."),
        Sample(label="Festive offer", kind="typical",
               text="Diwali sale at Sharma Electronics: 20% off on all fans and heaters until Sunday. Show this message at the counter."),
        Sample(label="Plain announcement", kind="legit",
               text="Our store timings are now 10 AM to 8 PM, Monday to Saturday."),
    ],
    llm_context="Judge how likely each person is to notice, like, try, buy, and share a product message. Many people ignore ads; interest and trust matter most.",
)

# --------------------------------------------------------------------------
# PACK 4: civic and policy (beneficial)
# --------------------------------------------------------------------------
CIVIC_POLICY = DomainPack(
    id="civic_policy",
    title="Civic and policy notice",
    description="How an official notice (new rule, deadline, scheme enrolment) reaches residents and gets followed.",
    act_label="complies or enrols",
    valence="beneficial",
    allow_message_improvement=True,
    base_rates={"engage": 0.20, "act": 0.07, "share": 0.06, "report": 0.01},
    trait_weights={
        "engage": {"attention": 0.25, "trust_in_source": 0.20, "domain_interest": 0.25,
                   "language_barrier": -0.25},
        "act":    {"trust_in_source": 0.25, "digital_literacy": 0.15, "language_barrier": -0.25,
                   "time_pressure": -0.20, "financial_cushion": 0.10},
        "share":  {"social_sharing": 0.30, "domain_interest": 0.20, "trust_in_source": 0.10},
        "report": {"skepticism": 0.10},
    },
    feature_weights={
        "engage": {"authority_claim": 0.15, "urgency": 0.10},
        "act":    {"call_to_action": 0.30, "urgency": 0.15, "authority_claim": 0.10, "has_link": -0.10},
        "share":  {"emotional_pull": 0.10, "urgency": 0.05},
        "report": {},
    },
    tier_thresholds=TierThresholds(reach=[0.15, 0.35, 0.60], r0=[0.6, 0.9, 1.2]),
    verdict_labels=["Unnoticed", "Patchy reach", "Good reach", "Universal"],
    audience_presets=[P_HOMEMAKERS, P_SHOPKEEPERS, P_FARMERS],
    samples=[
        Sample(label="Scheme enrolment deadline", kind="typical",
               text="Last date to enrol for the crop insurance scheme is 31 October. Visit your nearest bank branch or CSC with land records and Aadhaar."),
        Sample(label="Waste segregation rule", kind="typical",
               text="From 1 November, separate wet and dry waste at home. Unsegregated waste will not be collected. Bins will be provided free."),
        Sample(label="General reminder", kind="legit",
               text="Municipal office will remain closed on Saturday for maintenance work."),
    ],
    llm_context="Judge how likely each person is to notice, understand, follow, and pass on an official notice. Consider language, trust in authority, effort required, and deadlines.",
)

# --------------------------------------------------------------------------
# PACK 5: misinformation (harmful)
# --------------------------------------------------------------------------
MISINFORMATION = DomainPack(
    id="misinformation",
    title="Misinformation and rumours",
    description="How a false or unverified claim spreads, who believes it, and whether a correction can slow it. Defensive analysis only.",
    act_label="believes and reshares",
    valence="harmful",
    allow_message_improvement=False,
    base_rates={"engage": 0.22, "act": 0.06, "share": 0.10, "report": 0.03},
    trait_weights={
        "engage": {"attention": 0.20, "emotional_reactivity": 0.30, "skepticism": -0.25,
                   "social_sharing": 0.10},
        "act":    {"trust_in_source": 0.15, "skepticism": -0.40, "emotional_reactivity": 0.25,
                   "digital_literacy": -0.15, "domain_interest": 0.10},
        "share":  {"social_sharing": 0.40, "emotional_reactivity": 0.25, "skepticism": -0.25,
                   "time_pressure": 0.10},
        "report": {"skepticism": 0.30, "digital_literacy": 0.20},
    },
    feature_weights={
        "engage": {"emotional_pull": 0.30, "urgency": 0.15, "authority_claim": 0.10},
        "act":    {"emotional_pull": 0.20, "authority_claim": 0.15, "urgency": 0.10},
        "share":  {"emotional_pull": 0.30, "urgency": 0.20, "call_to_action": 0.15},
        "report": {},
    },
    tier_thresholds=TierThresholds(reach=[0.10, 0.30, 0.55], r0=[0.7, 1.0, 1.4]),
    verdict_labels=["Fizzles out", "Local rumour", "Widely believed", "Viral falsehood"],
    audience_presets=[P_HOMEMAKERS, P_STUDENTS, P_SENIORS],
    samples=[
        Sample(label="Home-remedy rumour (simulation sample)", kind="typical",
               text="FORWARDED: Doctors say drinking hot lemon water every hour cures dengue in a day. Share with everyone you love!"),
        Sample(label="ATM hoax (simulation sample)", kind="typical",
               text="URGENT: All ATMs will be shut for 3 days from tomorrow. Withdraw cash today and forward to your family."),
        Sample(label="Official clarification", kind="legit",
               text="Clarification: there is no ATM shutdown planned. All bank services are operating normally. Source: official bank notice."),
    ],
    llm_context="Judge how likely each person is to believe, reshare, doubt, or flag a possibly false claim. Strong emotion and social proof raise sharing; verification habits lower it.",
)

# --------------------------------------------------------------------------
# Registry and helpers
# --------------------------------------------------------------------------
PACKS: Dict[str, DomainPack] = {
    p.id: p for p in [SCAM_FRAUD, HEALTH_CAMPAIGN, PRODUCT_MARKETING, CIVIC_POLICY, MISINFORMATION]
}


def get_pack(pack_id: str) -> DomainPack:
    if pack_id not in PACKS:
        raise KeyError(f"Unknown pack '{pack_id}'. Available: {list(PACKS)}")
    return PACKS[pack_id]


def public_view(pack: DomainPack) -> dict:
    """What GET /packs returns: no internal weights."""
    return {
        "id": pack.id,
        "title": pack.title,
        "description": pack.description,
        "act_label": pack.act_label,
        "valence": pack.valence,
        "allow_message_improvement": pack.allow_message_improvement,
        "verdict_labels": pack.verdict_labels,
        "audience_presets": [p.model_dump() for p in pack.audience_presets],
        "samples": [s.model_dump() for s in pack.samples],
    }


def list_public_packs() -> List[dict]:
    return [public_view(p) for p in PACKS.values()]


def verdict_for(pack: DomainPack, reach: float, r0: float) -> dict:
    """Return {tier: 0-3, label}. Tier is the higher of the reach tier and R0 tier."""
    def tier(value: float, cuts: List[float]) -> int:
        return sum(value >= c for c in cuts)
    t = max(tier(reach, pack.tier_thresholds.reach), tier(r0, pack.tier_thresholds.r0))
    return {"tier": t, "label": pack.verdict_labels[t]}


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def base_probabilities(pack: DomainPack, traits: Dict[str, float],
                       features: Dict[str, float]) -> Dict[str, float]:
    """
    Rule-based baseline reaction used (a) as the LLM fallback and
    (b) as the 40% rule component in the hybrid blend.

    logit(p_action) = logit(base)
                    + TRAIT_K   * sum(trait_weight   * (trait - 0.5))
                    + FEATURE_K * sum(feature_weight * feature)
    Ordering is enforced afterwards: act <= engage.
    """
    out: Dict[str, float] = {}
    for a in ACTIONS:
        z = _logit(pack.base_rates[a])
        z += TRAIT_K * sum(w * (traits.get(t, 0.5) - 0.5)
                           for t, w in pack.trait_weights.get(a, {}).items())
        z += FEATURE_K * sum(w * features.get(f, 0.0)
                             for f, w in pack.feature_weights.get(a, {}).items())
        out[a] = _sigmoid(z)
    out["act"] = min(out["act"], out["engage"])
    return out


if __name__ == "__main__":
    # Self-check: python backend/domain_packs.py
    for pid, pack in PACKS.items():
        print(f"[ok] {pid:18s} valence={pack.valence:10s} presets={[p.id for p in pack.audience_presets]}")
    vuln = {t: 0.5 for t in TRAITS} | {"skepticism": 0.2, "digital_literacy": 0.25,
                                        "trust_in_source": 0.8, "time_pressure": 0.7}
    skep = {t: 0.5 for t in TRAITS} | {"skepticism": 0.9, "digital_literacy": 0.8}
    feats = {"urgency": 0.9, "emotional_pull": 0.6, "authority_claim": 0.8,
             "call_to_action": 0.9, "has_link": 1.0}
    pv = base_probabilities(SCAM_FRAUD, vuln, feats)
    ps = base_probabilities(SCAM_FRAUD, skep, feats)
    print("scam act  vulnerable:", round(pv["act"], 3), " skeptical:", round(ps["act"], 3))
    assert pv["act"] > ps["act"], "vulnerable users should fall more than skeptics"
    legit_feats = {"urgency": 0.1, "emotional_pull": 0.1, "authority_claim": 0.3,
                   "call_to_action": 0.1, "has_link": 0.0}
    pl = base_probabilities(SCAM_FRAUD, vuln, legit_feats)
    print("legit alert act (vulnerable):", round(pl["act"], 3))
    assert pv["act"] > 2 * pl["act"], "scam should score well above a legit alert"
    assert ps["act"] < 0.15, "very sceptical people should rarely act"
    assert pl["act"] < 0.20, "a genuine alert should rarely trigger action"
    for pid, pk in PACKS.items():
        mid = {t: 0.5 for t in TRAITS}
        pr = base_probabilities(pk, mid, {f: 0.5 for f in FEATURES})
        assert all(0 <= v <= 1 for v in pr.values()) and pr["act"] <= pr["engage"], pid
        print(f"  {pid:18s} average person, mid-strength message:",
              {k: round(v, 2) for k, v in pr.items()})
    print("verdict example:", verdict_for(SCAM_FRAUD, reach=0.4, r0=1.2))
