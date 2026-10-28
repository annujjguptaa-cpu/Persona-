"""Core Pydantic models for Persona: Synthetic Audience Message Spread Simulator.

All probabilities are floats bounded in [0.0, 1.0] validated with Field(ge=0, le=1).
"""

from typing import Any, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PersonaBaseModel(BaseModel):
    """Base model allowing flexible forward-compatible extensions."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class Archetype(PersonaBaseModel):
    """Represents a psychological/behavioral persona archetype in the synthetic audience.

    - id: Unique identifier (e.g. 'skeptic', 'amplifying_influencer', 'anxious_parent')
    - label: Human-readable name
    - description: Behavioral synopsis
    - share_of_population: Probability / fraction of total population [0.0, 1.0]
    - traits: Baseline behavioral traits (e.g. {'skepticism': 0.8, 'credulity': 0.2})
    """

    id: str
    label: str
    description: str
    share_of_population: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of population represented as probability float between 0 and 1",
    )
    traits: dict[str, float] = Field(
        default_factory=dict,
        description="Behavioral traits with intensity floats",
    )


class Person(PersonaBaseModel):
    """An individual synthetic agent in the social graph.

    - id: Unique node ID
    - name: Generated persona name
    - archetype_id: Reference to Archetype ID
    - age: Person's age
    - language: Primary language code or label
    - area: Geographic/demographic context (e.g. 'metro', 'suburban', 'rural')
    - traits: Individual traits individualized from archetype baseline
    """

    id: str
    name: str
    archetype_id: str
    age: int = Field(default=30, ge=0, description="Age in years")
    language: str = Field(default="en", description="Primary language")
    area: str = Field(default="metro", description="Geographic or demographic area")
    traits: dict[str, float] = Field(
        default_factory=dict,
        description="Trait modifiers for this person",
    )


class MessageFeatures(PersonaBaseModel):
    """Extracted behavioral and structural features of a candidate message.

    Probabilities and intensity ratings are floats bounded in [0.0, 1.0].
    """

    is_suspicious: bool = Field(
        default=False,
        description="Whether message contains scam, deceptive, or misleading cues",
    )
    urgency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Urgency intensity score between 0 and 1",
    )
    emotional_pull: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Emotional pull intensity score between 0 and 1",
    )
    authority_claim: Union[str, float] = Field(
        default="none",
        description="Degree of authority claim (category from ['none', 'bank', 'insurer', 'hospital', 'govt', 'brand', 'other'] or score 0-1)",
    )
    call_to_action: Union[bool, float] = Field(
        default=False,
        description="Presence or strength of call-to-action (boolean or score 0-1)",
    )
    has_link: bool = Field(
        default=False,
        description="Whether message embeds URLs or shortlinks",
    )
    language: str = Field(
        default="en",
        description="Detected language code",
    )
    channel_guess: str = Field(
        default="unknown",
        description="Target medium (e.g. 'whatsapp', 'sms', 'email', 'x_twitter')",
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Identified suspicious cues or hazards",
    )
    legit_signals: list[str] = Field(
        default_factory=list,
        description="Identified authenticating signals or verifiable references",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Core subject themes (e.g. 'banking', 'health', 'politics')",
    )

    @field_validator("authority_claim", mode="before")
    @classmethod
    def validate_authority_claim(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if v < 0.0 or v > 1.0:
                raise ValueError("authority_claim numeric score must be between 0.0 and 1.0")
        elif isinstance(v, str):
            allowed = {"none", "bank", "insurer", "hospital", "govt", "brand", "other"}
            s = v.strip().lower()
            if s in allowed:
                return s
            try:
                f_val = float(s)
                if f_val < 0.0 or f_val > 1.0:
                    raise ValueError("authority_claim numeric score must be between 0.0 and 1.0")
                return f_val
            except ValueError:
                return "other" if s else "none"
        return v

    @property
    def authority_score(self) -> float:
        """Numeric strength score for authority claim in [0.0, 1.0]."""
        if isinstance(self.authority_claim, (int, float)):
            return float(self.authority_claim)
        weights = {
            "none": 0.0,
            "other": 0.35,
            "brand": 0.50,
            "hospital": 0.75,
            "insurer": 0.80,
            "bank": 0.85,
            "govt": 0.95,
        }
        return weights.get(str(self.authority_claim).lower(), 0.0)

    @field_validator("call_to_action", mode="before")
    @classmethod
    def validate_call_to_action(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if v < 0.0 or v > 1.0:
                raise ValueError("call_to_action numeric score must be between 0.0 and 1.0")
        return v


class Reaction(PersonaBaseModel):
    """Probabilistic reaction profile of an archetype exposed to a message.

    All response probabilities (ignore, notice, engage, act, share, report)
    are floats between 0.0 and 1.0 inclusive.
    """

    archetype_id: str
    ignore: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of ignoring the message [0.0, 1.0]",
    )
    notice: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of reading/noticing the message [0.0, 1.0]",
    )
    engage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of reacting/commenting/discussing [0.0, 1.0]",
    )
    act: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of following the call-to-action [0.0, 1.0]",
    )
    share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of propagating/forwarding to contacts [0.0, 1.0]",
    )
    report: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of flagging/reporting as abuse or spam [0.0, 1.0]",
    )
    top_trigger: str = Field(
        default="",
        description="Primary psychological trigger driving this reaction",
    )
    reason: str = Field(
        default="",
        description="Natural language rationale explaining archetype decision",
    )


class WaveState(PersonaBaseModel):
    """State of simulation at a specific diffusion wave/time-step.

    Captures nodes reached, reactions recorded, and propagation counts.
    """

    wave: int = Field(default=0, ge=0, description="Wave index starting from 0")
    reached_ids: list[str] = Field(
        default_factory=list,
        description="List of person node IDs reached during this wave",
    )
    newly_reached: int = Field(
        default=0,
        ge=0,
        description="Count of nodes receiving the message for the first time",
    )
    cumulative_reached: int = Field(
        default=0,
        ge=0,
        description="Total unique individuals exposed up to this wave",
    )
    reactions: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregated reaction tallies or maps for this wave",
    )
    shares: int = Field(
        default=0,
        ge=0,
        description="Number of shares triggered in this wave",
    )
    acts: int = Field(
        default=0,
        ge=0,
        description="Number of conversions / actions taken in this wave",
    )
    reports: int = Field(
        default=0,
        ge=0,
        description="Number of spam/abuse reports filed in this wave",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual telemetry for the wave",
    )


class SimulationResult(PersonaBaseModel):
    """Complete output payload from a simulated message cascade.

    - waves: Chronological sequence of WaveState snapshots
    - edges: Network propagation edges traversed during spread
    - totals: High-level final counts (reached, shared, acted, reported)
    - group_stats: Granular breakdowns across archetypes and demographics
    - verdict: Final executive verdict or defense assessment
    - insights: Key strategic takeaways from the diffusion dynamics
    """

    waves: list[WaveState] = Field(
        default_factory=list,
        description="Chronological diffusion wave snapshots",
    )
    edges: list[Any] = Field(
        default_factory=list,
        description="Network propagation paths and transmission edges",
    )
    totals: dict[str, Any] = Field(
        default_factory=dict,
        description="High-level cascade totals (reach, conversions, virality)",
    )
    group_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Subgroup performance and vulnerability distribution",
    )
    verdict: str = Field(
        default="",
        description="Summary verdict assessing viral potential or hazard",
    )
    insights: list[str] = Field(
        default_factory=list,
        description="Strategic observations and narrative takeaways",
    )
    node_states: dict[str, str] = Field(
        default_factory=dict,
        description="Sampled state per agent node ID (ignore, notice only, engage, act, share, report)",
    )
