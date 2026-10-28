"""
archetypes.py  -  Persona
=========================
Hand-curated Indian demographic and behavioral audience presets,
plus LLM-assisted generation of custom audience segments.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.cache import get_cache, set_cache, hash_key
from backend.domain_packs import TRAITS
from backend.llm import query_ollama
from backend.models import Archetype

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# JSON Schema for Ollama generation
# --------------------------------------------------------------------------
ARCHETYPES_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "archetypes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "share_of_population": {"type": "number"},
                    "traits": {
                        "type": "object",
                        "properties": {
                            "attention": {"type": "number"},
                            "digital_literacy": {"type": "number"},
                            "trust_in_source": {"type": "number"},
                            "skepticism": {"type": "number"},
                            "emotional_reactivity": {"type": "number"},
                            "social_sharing": {"type": "number"},
                            "time_pressure": {"type": "number"},
                            "domain_interest": {"type": "number"},
                            "financial_cushion": {"type": "number"},
                            "language_barrier": {"type": "number"},
                        },
                        "required": [
                            "attention",
                            "digital_literacy",
                            "trust_in_source",
                            "skepticism",
                            "emotional_reactivity",
                            "social_sharing",
                            "time_pressure",
                            "domain_interest",
                            "financial_cushion",
                            "language_barrier",
                        ],
                    },
                },
                "required": ["id", "label", "description", "share_of_population", "traits"],
            },
        }
    },
    "required": ["archetypes"],
}


# --------------------------------------------------------------------------
# PRESETS: 9 Audience Presets with 10 hand-written Indian Archetypes each
# --------------------------------------------------------------------------
PRESETS: Dict[str, List[Archetype]] = {
    # 1. Urban Seniors
    "urban_seniors": [
        Archetype(
            id="pensioner_cautious",
            label="Retired Govt Pensioner",
            description="Ex-central govt employee, relies on physical bank branch and pension passbook updates, highly cautious of online transactions.",
            share_of_population=0.12,
            traits={"attention": 0.70, "digital_literacy": 0.35, "trust_in_source": 0.85, "skepticism": 0.55, "emotional_reactivity": 0.30, "social_sharing": 0.40, "time_pressure": 0.15, "domain_interest": 0.65, "financial_cushion": 0.65, "language_barrier": 0.10},
        ),
        Archetype(
            id="whatsapp_uncle",
            label="WhatsApp Family Admin",
            description="Loves greeting forwards, devotional messages, and patriotic updates; broadcasts widely to family groups without verifying.",
            share_of_population=0.14,
            traits={"attention": 0.80, "digital_literacy": 0.45, "trust_in_source": 0.75, "skepticism": 0.25, "emotional_reactivity": 0.75, "social_sharing": 0.90, "time_pressure": 0.20, "domain_interest": 0.70, "financial_cushion": 0.60, "language_barrier": 0.15},
        ),
        Archetype(
            id="diaspora_parent",
            label="NRI Parent",
            description="Lives alone while children work abroad; comfortable with WhatsApp video calls but struggles with banking OTPs and UPI links.",
            share_of_population=0.10,
            traits={"attention": 0.60, "digital_literacy": 0.38, "trust_in_source": 0.70, "skepticism": 0.40, "emotional_reactivity": 0.60, "social_sharing": 0.50, "time_pressure": 0.25, "domain_interest": 0.50, "financial_cushion": 0.80, "language_barrier": 0.10},
        ),
        Archetype(
            id="society_secretary",
            label="RWA Management Secretary",
            description="Vigilant neighborhood elder, verifies circulars with local police and municipal ward officers, high skepticism of unverified notices.",
            share_of_population=0.08,
            traits={"attention": 0.85, "digital_literacy": 0.60, "trust_in_source": 0.60, "skepticism": 0.85, "emotional_reactivity": 0.25, "social_sharing": 0.65, "time_pressure": 0.40, "domain_interest": 0.80, "financial_cushion": 0.70, "language_barrier": 0.05},
        ),
        Archetype(
            id="devotional_matriarch",
            label="Devotional Matriarch",
            description="Spends mornings watching temple live-streams and satsangs; deeply moved by charitable and temple renovation appeals.",
            share_of_population=0.11,
            traits={"attention": 0.65, "digital_literacy": 0.30, "trust_in_source": 0.80, "skepticism": 0.20, "emotional_reactivity": 0.80, "social_sharing": 0.70, "time_pressure": 0.20, "domain_interest": 0.65, "financial_cushion": 0.50, "language_barrier": 0.25},
        ),
        Archetype(
            id="frugal_widow",
            label="Frugal Fixed-Deposit Saver",
            description="Living on modest fixed deposit interest; terrified of account freezing or unauthorized debits, susceptible to urgency panic.",
            share_of_population=0.10,
            traits={"attention": 0.55, "digital_literacy": 0.20, "trust_in_source": 0.70, "skepticism": 0.35, "emotional_reactivity": 0.70, "social_sharing": 0.20, "time_pressure": 0.20, "domain_interest": 0.60, "financial_cushion": 0.35, "language_barrier": 0.30},
        ),
        Archetype(
            id="tech_curious_grandfather",
            label="Tech-Curious Grandfather",
            description="Uses tablet for online chess, YouTube audiobooks, and e-newspapers; willing to explore apps but asks grandson before payments.",
            share_of_population=0.09,
            traits={"attention": 0.75, "digital_literacy": 0.55, "trust_in_source": 0.50, "skepticism": 0.65, "emotional_reactivity": 0.35, "social_sharing": 0.45, "time_pressure": 0.15, "domain_interest": 0.75, "financial_cushion": 0.60, "language_barrier": 0.10},
        ),
        Archetype(
            id="lonely_grandparent",
            label="Isolated Solitary Senior",
            description="Eager for social interaction, answers unknown callers gladly, vulnerable to conversational romance and lottery frauds.",
            share_of_population=0.08,
            traits={"attention": 0.50, "digital_literacy": 0.25, "trust_in_source": 0.85, "skepticism": 0.20, "emotional_reactivity": 0.85, "social_sharing": 0.50, "time_pressure": 0.10, "domain_interest": 0.45, "financial_cushion": 0.45, "language_barrier": 0.20},
        ),
        Archetype(
            id="retired_teacher",
            label="Retired School Teacher",
            description="Grammar stickler, reads fine print methodically, debunks sensationalist forwarded claims in neighborhood groups.",
            share_of_population=0.10,
            traits={"attention": 0.85, "digital_literacy": 0.50, "trust_in_source": 0.50, "skepticism": 0.80, "emotional_reactivity": 0.20, "social_sharing": 0.55, "time_pressure": 0.20, "domain_interest": 0.70, "financial_cushion": 0.55, "language_barrier": 0.05},
        ),
        Archetype(
            id="health_anxious_elder",
            label="Health-Anxious Chronic Patient",
            description="Manages diabetes and hypertension; eager to try Ayurvedic or herbal cures, easily drawn by promises of instant wellness.",
            share_of_population=0.08,
            traits={"attention": 0.75, "digital_literacy": 0.35, "trust_in_source": 0.65, "skepticism": 0.30, "emotional_reactivity": 0.75, "social_sharing": 0.60, "time_pressure": 0.25, "domain_interest": 0.90, "financial_cushion": 0.50, "language_barrier": 0.20},
        ),
    ],

    # 2. College Students
    "college_students": [
        Archetype(
            id="budget_hustler",
            label="Gig & Intern Hustler",
            description="Undergrad looking for paid internships, freelance gigs, and cashbacks; low savings, fast to sign up for earning apps.",
            share_of_population=0.15,
            traits={"attention": 0.80, "digital_literacy": 0.80, "trust_in_source": 0.35, "skepticism": 0.45, "emotional_reactivity": 0.60, "social_sharing": 0.70, "time_pressure": 0.65, "domain_interest": 0.85, "financial_cushion": 0.20, "language_barrier": 0.05},
        ),
        Archetype(
            id="exam_crammer",
            label="Stressed Exam Crammer",
            description="Buried under notes and entrance exams; constantly on Telegram channels hunting for leaked question papers and shortcuts.",
            share_of_population=0.12,
            traits={"attention": 0.65, "digital_literacy": 0.38, "trust_in_source": 0.55, "skepticism": 0.35, "emotional_reactivity": 0.80, "social_sharing": 0.60, "time_pressure": 0.90, "domain_interest": 0.90, "financial_cushion": 0.30, "language_barrier": 0.15},
        ),
        Archetype(
            id="campus_influencer",
            label="Campus Meme Influencer",
            description="Active on Instagram Reels and campus gossip channels; forwards provocative or funny posts instantly for social clout.",
            share_of_population=0.10,
            traits={"attention": 0.85, "digital_literacy": 0.85, "trust_in_source": 0.30, "skepticism": 0.30, "emotional_reactivity": 0.70, "social_sharing": 0.95, "time_pressure": 0.50, "domain_interest": 0.75, "financial_cushion": 0.40, "language_barrier": 0.05},
        ),
        Archetype(
            id="skeptical_coder",
            label="CS Major Tech Skeptic",
            description="Inspects URL domains, checks SSL certificates and permissions, exposes fake giveaways and spam bots in batch groups.",
            share_of_population=0.10,
            traits={"attention": 0.90, "digital_literacy": 0.95, "trust_in_source": 0.20, "skepticism": 0.88, "emotional_reactivity": 0.20, "social_sharing": 0.40, "time_pressure": 0.40, "domain_interest": 0.80, "financial_cushion": 0.35, "language_barrier": 0.02},
        ),
        Archetype(
            id="fomo_socialite",
            label="FOMO Socialite",
            description="Afraid of missing out on concerts, clubs, and parties; quick to click event passes and referral contest invitations.",
            share_of_population=0.11,
            traits={"attention": 0.75, "digital_literacy": 0.75, "trust_in_source": 0.50, "skepticism": 0.30, "emotional_reactivity": 0.75, "social_sharing": 0.85, "time_pressure": 0.55, "domain_interest": 0.80, "financial_cushion": 0.50, "language_barrier": 0.05},
        ),
        Archetype(
            id="small_town_fresher",
            label="Small-Town Fresher",
            description="First time studying in a Tier-1 metro; unfamiliar with local slangs and rapid scam formats, trusts senior recommendations.",
            share_of_population=0.10,
            traits={"attention": 0.60, "digital_literacy": 0.35, "trust_in_source": 0.70, "skepticism": 0.40, "emotional_reactivity": 0.65, "social_sharing": 0.50, "time_pressure": 0.50, "domain_interest": 0.70, "financial_cushion": 0.25, "language_barrier": 0.35},
        ),
        Archetype(
            id="competitive_aspirant",
            label="Civil Services Aspirant",
            description="Prepares for UPSC/State PSC; strictly verifies fact-checks, policy updates, and notifications using official PIB releases.",
            share_of_population=0.09,
            traits={"attention": 0.90, "digital_literacy": 0.70, "trust_in_source": 0.65, "skepticism": 0.75, "emotional_reactivity": 0.25, "social_sharing": 0.35, "time_pressure": 0.70, "domain_interest": 0.85, "financial_cushion": 0.35, "language_barrier": 0.10},
        ),
        Archetype(
            id="gaming_streamer",
            label="Mobile Gamer & Streamer",
            description="Spends hours on BGMI/Free Fire; responsive to game skin codes, tournament entries, and UPI wallet bonuses.",
            share_of_population=0.08,
            traits={"attention": 0.85, "digital_literacy": 0.80, "trust_in_source": 0.40, "skepticism": 0.45, "emotional_reactivity": 0.65, "social_sharing": 0.75, "time_pressure": 0.40, "domain_interest": 0.90, "financial_cushion": 0.30, "language_barrier": 0.08},
        ),
        Archetype(
            id="student_activist",
            label="Student Union Representative",
            description="Passionate about campus policies, fee increases, and youth employment; mobilizes student community over WhatsApp broadcasts.",
            share_of_population=0.08,
            traits={"attention": 0.85, "digital_literacy": 0.70, "trust_in_source": 0.30, "skepticism": 0.55, "emotional_reactivity": 0.85, "social_sharing": 0.90, "time_pressure": 0.60, "domain_interest": 0.85, "financial_cushion": 0.30, "language_barrier": 0.05},
        ),
        Archetype(
            id="quiet_day_scholar",
            label="Commuter Day Scholar",
            description="Commutes 2 hours daily on public buses; listens to music or study audio, filters out college drama and spam notifications.",
            share_of_population=0.07,
            traits={"attention": 0.50, "digital_literacy": 0.60, "trust_in_source": 0.45, "skepticism": 0.65, "emotional_reactivity": 0.30, "social_sharing": 0.30, "time_pressure": 0.70, "domain_interest": 0.50, "financial_cushion": 0.35, "language_barrier": 0.10},
        ),
    ],

    # 3. Rural Farmers
    "rural_farmers": [
        Archetype(
            id="progressive_grower",
            label="Progressive Drip Farmer",
            description="Cultivates cash crops, tracks mandi prices on smartphone apps, receptive to modern agtech and certified seed programs.",
            share_of_population=0.12,
            traits={"attention": 0.75, "digital_literacy": 0.65, "trust_in_source": 0.65, "skepticism": 0.55, "emotional_reactivity": 0.40, "social_sharing": 0.60, "time_pressure": 0.45, "domain_interest": 0.90, "financial_cushion": 0.60, "language_barrier": 0.20},
        ),
        Archetype(
            id="elderly_smallholder",
            label="Elderly Marginal Farmer",
            description="Owns 1-2 acres, uses basic keypad phone; relies on village CSC center or grandson for PM-KISAN subsidy registration.",
            share_of_population=0.14,
            traits={"attention": 0.45, "digital_literacy": 0.15, "trust_in_source": 0.80, "skepticism": 0.35, "emotional_reactivity": 0.60, "social_sharing": 0.40, "time_pressure": 0.30, "domain_interest": 0.80, "financial_cushion": 0.20, "language_barrier": 0.55},
        ),
        Archetype(
            id="debt_stressed_ryot",
            label="Debt-Stressed Monsoon Farmer",
            description="Vulnerable to crop failure and high-interest private debt; eagerly searches for loan waivers, subsidies, or emergency credit.",
            share_of_population=0.13,
            traits={"attention": 0.60, "digital_literacy": 0.30, "trust_in_source": 0.70, "skepticism": 0.25, "emotional_reactivity": 0.90, "social_sharing": 0.50, "time_pressure": 0.70, "domain_interest": 0.95, "financial_cushion": 0.10, "language_barrier": 0.40},
        ),
        Archetype(
            id="panchayat_influencer",
            label="Panchayat Ward Member",
            description="Influential village council representative; verifies government notices at Block office before sharing with village elders.",
            share_of_population=0.10,
            traits={"attention": 0.80, "digital_literacy": 0.55, "trust_in_source": 0.60, "skepticism": 0.70, "emotional_reactivity": 0.35, "social_sharing": 0.75, "time_pressure": 0.40, "domain_interest": 0.85, "financial_cushion": 0.55, "language_barrier": 0.15},
        ),
        Archetype(
            id="agri_input_dealer",
            label="Fertilizer & Seed Dealer",
            description="Runs village input shop and UPI payment point; community hub dispensing advice on crop diseases and weather forecasts.",
            share_of_population=0.09,
            traits={"attention": 0.85, "digital_literacy": 0.65, "trust_in_source": 0.50, "skepticism": 0.65, "emotional_reactivity": 0.30, "social_sharing": 0.70, "time_pressure": 0.50, "domain_interest": 0.90, "financial_cushion": 0.65, "language_barrier": 0.15},
        ),
        Archetype(
            id="young_returnee_farmer",
            label="Tech-Savvy Returnee Farmer",
            description="Returned from city to innovate in organic farming; watches YouTube tutorials, questions spurious chemical claims.",
            share_of_population=0.09,
            traits={"attention": 0.85, "digital_literacy": 0.80, "trust_in_source": 0.35, "skepticism": 0.75, "emotional_reactivity": 0.30, "social_sharing": 0.65, "time_pressure": 0.40, "domain_interest": 0.85, "financial_cushion": 0.50, "language_barrier": 0.10},
        ),
        Archetype(
            id="sharecropper_tenant",
            label="Landless Tenant Laborer",
            description="Works on seasonal leases; semi-literate in regional script, relies on oral communication and local rumor networks.",
            share_of_population=0.11,
            traits={"attention": 0.40, "digital_literacy": 0.10, "trust_in_source": 0.80, "skepticism": 0.30, "emotional_reactivity": 0.75, "social_sharing": 0.35, "time_pressure": 0.60, "domain_interest": 0.75, "financial_cushion": 0.10, "language_barrier": 0.60},
        ),
        Archetype(
            id="cooperative_dairy_member",
            label="Dairy Cooperative Supplier",
            description="Supplies milk twice daily to cooperative center; familiar with SMS balance alerts and cattle vaccination campaigns.",
            share_of_population=0.08,
            traits={"attention": 0.65, "digital_literacy": 0.45, "trust_in_source": 0.75, "skepticism": 0.45, "emotional_reactivity": 0.45, "social_sharing": 0.50, "time_pressure": 0.50, "domain_interest": 0.80, "financial_cushion": 0.40, "language_barrier": 0.30},
        ),
        Archetype(
            id="subsidy_tracker",
            label="Solar & Equipment Subsidy Seeker",
            description="Constantly watches for government tractor or drip irrigation subsidies; prone to clicking fake portal links promising fast funds.",
            share_of_population=0.08,
            traits={"attention": 0.75, "digital_literacy": 0.45, "trust_in_source": 0.65, "skepticism": 0.35, "emotional_reactivity": 0.70, "social_sharing": 0.60, "time_pressure": 0.55, "domain_interest": 0.90, "financial_cushion": 0.35, "language_barrier": 0.25},
        ),
        Archetype(
            id="cautious_traditionalist",
            label="Traditional Village Elder",
            description="Skeptical of digital apps, private bank agents, and non-local schemes; warns fellow villagers against clicking links.",
            share_of_population=0.06,
            traits={"attention": 0.60, "digital_literacy": 0.20, "trust_in_source": 0.40, "skepticism": 0.75, "emotional_reactivity": 0.40, "social_sharing": 0.45, "time_pressure": 0.20, "domain_interest": 0.65, "financial_cushion": 0.45, "language_barrier": 0.50},
        ),
    ],

    # 4. Working Parents
    "working_parents": [
        Archetype(
            id="overloaded_dual_career",
            label="Overloaded Dual-Career Parent",
            description="Juggles corporate sprints and child care; high time stress, acts quickly on school messages without full scrutiny.",
            share_of_population=0.14,
            traits={"attention": 0.60, "digital_literacy": 0.85, "trust_in_source": 0.50, "skepticism": 0.45, "emotional_reactivity": 0.65, "social_sharing": 0.60, "time_pressure": 0.90, "domain_interest": 0.85, "financial_cushion": 0.70, "language_barrier": 0.05},
        ),
        Archetype(
            id="school_whatsapp_admin",
            label="School Class WhatsApp Admin",
            description="Coordinates parents for school trips and homework alerts; highly responsive and amplifying on child health and safety issues.",
            share_of_population=0.11,
            traits={"attention": 0.85, "digital_literacy": 0.70, "trust_in_source": 0.60, "skepticism": 0.40, "emotional_reactivity": 0.75, "social_sharing": 0.90, "time_pressure": 0.60, "domain_interest": 0.90, "financial_cushion": 0.65, "language_barrier": 0.05},
        ),
        Archetype(
            id="single_working_mother",
            label="Single Working Mother",
            description="Sole breadwinner managing household budget and tuition fees; price-sensitive, highly protective against financial waste.",
            share_of_population=0.09,
            traits={"attention": 0.75, "digital_literacy": 0.65, "trust_in_source": 0.45, "skepticism": 0.65, "emotional_reactivity": 0.60, "social_sharing": 0.45, "time_pressure": 0.85, "domain_interest": 0.80, "financial_cushion": 0.35, "language_barrier": 0.10},
        ),
        Archetype(
            id="nutrition_conscious_mom",
            label="Holistic Nutrition Researcher",
            description="Researches organic ingredients, pediatrician recommendations, and pediatric vaccines; skeptical of sponsored chemical ads.",
            share_of_population=0.11,
            traits={"attention": 0.85, "digital_literacy": 0.75, "trust_in_source": 0.40, "skepticism": 0.70, "emotional_reactivity": 0.50, "social_sharing": 0.70, "time_pressure": 0.55, "domain_interest": 0.95, "financial_cushion": 0.60, "language_barrier": 0.05},
        ),
        Archetype(
            id="commuter_father",
            label="Suburban Commuter Father",
            description="Spends 2.5 hours on suburban trains or cabs; skims news and family chats in transit, moderate attention span.",
            share_of_population=0.12,
            traits={"attention": 0.50, "digital_literacy": 0.70, "trust_in_source": 0.55, "skepticism": 0.50, "emotional_reactivity": 0.50, "social_sharing": 0.50, "time_pressure": 0.80, "domain_interest": 0.65, "financial_cushion": 0.60, "language_barrier": 0.08},
        ),
        Archetype(
            id="family_finance_planner",
            label="Prudent Family Investor",
            description="Systematic investor in child education funds and health insurance; verifies authenticity of policy and tax documents.",
            share_of_population=0.09,
            traits={"attention": 0.80, "digital_literacy": 0.75, "trust_in_source": 0.50, "skepticism": 0.75, "emotional_reactivity": 0.30, "social_sharing": 0.35, "time_pressure": 0.50, "domain_interest": 0.85, "financial_cushion": 0.75, "language_barrier": 0.05},
        ),
        Archetype(
            id="first_time_parent",
            label="Anxious First-Time Parent",
            description="New parent of an infant, highly anxious about pediatric infections or developmental milestones; easily triggered by scare alerts.",
            share_of_population=0.12,
            traits={"attention": 0.75, "digital_literacy": 0.38, "trust_in_source": 0.70, "skepticism": 0.30, "emotional_reactivity": 0.90, "social_sharing": 0.75, "time_pressure": 0.85, "domain_interest": 0.95, "financial_cushion": 0.50, "language_barrier": 0.10},
        ),
        Archetype(
            id="grandparent_co_carer",
            label="Multigenerational Caregiver",
            description="Supports both schooling children and aging in-laws; schedules clinical tests and vaccinations, vulnerable to medical scams.",
            share_of_population=0.08,
            traits={"attention": 0.65, "digital_literacy": 0.35, "trust_in_source": 0.75, "skepticism": 0.35, "emotional_reactivity": 0.75, "social_sharing": 0.65, "time_pressure": 0.80, "domain_interest": 0.85, "financial_cushion": 0.45, "language_barrier": 0.20},
        ),
        Archetype(
            id="neighborhood_coordinator",
            label="Community Activity Organizer",
            description="Plans society weekend tuition batches and sports coaching; well-connected, active in propagating local community notices.",
            share_of_population=0.08,
            traits={"attention": 0.75, "digital_literacy": 0.65, "trust_in_source": 0.55, "skepticism": 0.50, "emotional_reactivity": 0.55, "social_sharing": 0.85, "time_pressure": 0.60, "domain_interest": 0.80, "financial_cushion": 0.60, "language_barrier": 0.05},
        ),
        Archetype(
            id="skeptical_doctor_parent",
            label="Medical/Academic Professional Parent",
            description="Clinician or researcher parent; actively debunks unscientific medical forwards in school groups and demands peer-reviewed sources.",
            share_of_population=0.06,
            traits={"attention": 0.90, "digital_literacy": 0.85, "trust_in_source": 0.30, "skepticism": 0.90, "emotional_reactivity": 0.15, "social_sharing": 0.40, "time_pressure": 0.75, "domain_interest": 0.85, "financial_cushion": 0.85, "language_barrier": 0.02},
        ),
    ],

    # 5. Small Shopkeepers
    "small_shopkeepers": [
        Archetype(
            id="kirana_veteran",
            label="Neighborhood Kirana Owner",
            description="Runs family provisions store for 25 years; relies on steady customer relations, cautious with formal tax and digital audits.",
            share_of_population=0.13,
            traits={"attention": 0.70, "digital_literacy": 0.45, "trust_in_source": 0.55, "skepticism": 0.65, "emotional_reactivity": 0.35, "social_sharing": 0.50, "time_pressure": 0.60, "domain_interest": 0.80, "financial_cushion": 0.50, "language_barrier": 0.20},
        ),
        Archetype(
            id="soundbox_trader",
            label="Busy Soundbox Retailer",
            description="Runs busy sweet/chaat shop, relies strictly on Paytm/PhonePe audio alerts for payments; prone to fake audio notification spoofing.",
            share_of_population=0.14,
            traits={"attention": 0.55, "digital_literacy": 0.50, "trust_in_source": 0.65, "skepticism": 0.40, "emotional_reactivity": 0.55, "social_sharing": 0.45, "time_pressure": 0.80, "domain_interest": 0.75, "financial_cushion": 0.45, "language_barrier": 0.15},
        ),
        Archetype(
            id="wholesale_distributor",
            label="Wholesale Grain Merchant",
            description="Manages credit ledgers (khata) and bulk inventory; sensitive to supplier fraud and GST compliance notices.",
            share_of_population=0.11,
            traits={"attention": 0.80, "digital_literacy": 0.55, "trust_in_source": 0.50, "skepticism": 0.70, "emotional_reactivity": 0.30, "social_sharing": 0.40, "time_pressure": 0.60, "domain_interest": 0.85, "financial_cushion": 0.70, "language_barrier": 0.15},
        ),
        Archetype(
            id="tech_adopting_chemist",
            label="Digital Pharmacy Owner",
            description="Uses barcode billing software; verifies drug recall circulars and FDA notices with state chemists association.",
            share_of_population=0.10,
            traits={"attention": 0.85, "digital_literacy": 0.70, "trust_in_source": 0.60, "skepticism": 0.75, "emotional_reactivity": 0.25, "social_sharing": 0.60, "time_pressure": 0.50, "domain_interest": 0.90, "financial_cushion": 0.65, "language_barrier": 0.05},
        ),
        Archetype(
            id="struggling_kiosk_vendor",
            label="Tea Stall / Kiosk Vendor",
            description="Operates roadside tea/paan stall with a printed QR code; low profit margins, highly vulnerable to fraudulent payment reversals.",
            share_of_population=0.12,
            traits={"attention": 0.45, "digital_literacy": 0.25, "trust_in_source": 0.75, "skepticism": 0.30, "emotional_reactivity": 0.80, "social_sharing": 0.35, "time_pressure": 0.75, "domain_interest": 0.65, "financial_cushion": 0.15, "language_barrier": 0.35},
        ),
        Archetype(
            id="gst_anxious_dealer",
            label="Tax-Anxious Small Trader",
            description="Fears penalty notices from commercial taxes department; quickly reacts in panic to bogus tax seizure or audit SMS threats.",
            share_of_population=0.09,
            traits={"attention": 0.75, "digital_literacy": 0.35, "trust_in_source": 0.70, "skepticism": 0.35, "emotional_reactivity": 0.85, "social_sharing": 0.40, "time_pressure": 0.65, "domain_interest": 0.85, "financial_cushion": 0.45, "language_barrier": 0.25},
        ),
        Archetype(
            id="young_retail_successor",
            label="Next-Gen Retail Successor",
            description="Son/daughter taking over family garment shop; experiments with WhatsApp business catalogs, Instagram reels, and bulk SMS promos.",
            share_of_population=0.09,
            traits={"attention": 0.80, "digital_literacy": 0.80, "trust_in_source": 0.40, "skepticism": 0.50, "emotional_reactivity": 0.50, "social_sharing": 0.80, "time_pressure": 0.50, "domain_interest": 0.90, "financial_cushion": 0.55, "language_barrier": 0.05},
        ),
        Archetype(
            id="market_association_leader",
            label="Vyapar Mandal President",
            description="Leads local trade association; communicates municipal bandhs, license policies, and utility tariff changes across hundreds of shops.",
            share_of_population=0.08,
            traits={"attention": 0.85, "digital_literacy": 0.60, "trust_in_source": 0.50, "skepticism": 0.65, "emotional_reactivity": 0.40, "social_sharing": 0.90, "time_pressure": 0.50, "domain_interest": 0.90, "financial_cushion": 0.70, "language_barrier": 0.10},
        ),
        Archetype(
            id="frugal_repair_mechanic",
            label="Mobile & Electronics Repairer",
            description="Tech-literate with hardware and device firmware, but skeptical of unverified fintech credit apps and personal loan spam.",
            share_of_population=0.08,
            traits={"attention": 0.70, "digital_literacy": 0.70, "trust_in_source": 0.35, "skepticism": 0.70, "emotional_reactivity": 0.30, "social_sharing": 0.40, "time_pressure": 0.55, "domain_interest": 0.70, "financial_cushion": 0.35, "language_barrier": 0.15},
        ),
        Archetype(
            id="scam_burned_merchant",
            label="Previously Defrauded Merchant",
            description="Lost Rs 5,000 previously to a fake screenshot scam; now hyper-skeptical of any promo link, unfamiliar app, or payment discount.",
            share_of_population=0.06,
            traits={"attention": 0.85, "digital_literacy": 0.50, "trust_in_source": 0.20, "skepticism": 0.85, "emotional_reactivity": 0.45, "social_sharing": 0.35, "time_pressure": 0.45, "domain_interest": 0.70, "financial_cushion": 0.45, "language_barrier": 0.20},
        ),
    ],

    # 6. Community Health Workers
    "community_health_workers": [
        Archetype(
            id="veteran_asha_didi",
            label="Veteran ASHA Facilitator",
            description="15 years visiting every village home; deeply trusted by mothers, dispels vaccine and institutional delivery hesitations.",
            share_of_population=0.15,
            traits={"attention": 0.80, "digital_literacy": 0.38, "trust_in_source": 0.80, "skepticism": 0.55, "emotional_reactivity": 0.50, "social_sharing": 0.85, "time_pressure": 0.70, "domain_interest": 0.95, "financial_cushion": 0.25, "language_barrier": 0.30},
        ),
        Archetype(
            id="young_anganwadi_worker",
            label="Anganwadi Smartphone Worker",
            description="Enters child growth records into Poshan Tracker app; coordinates supplementary nutrition and early childhood education.",
            share_of_population=0.13,
            traits={"attention": 0.80, "digital_literacy": 0.60, "trust_in_source": 0.75, "skepticism": 0.50, "emotional_reactivity": 0.45, "social_sharing": 0.75, "time_pressure": 0.65, "domain_interest": 0.90, "financial_cushion": 0.30, "language_barrier": 0.15},
        ),
        Archetype(
            id="phc_staff_nurse",
            label="PHC Staff Nurse",
            description="Administers routine immunizations and maternal care; double-checks medical circulars against official Directorate protocols.",
            share_of_population=0.11,
            traits={"attention": 0.85, "digital_literacy": 0.65, "trust_in_source": 0.70, "skepticism": 0.65, "emotional_reactivity": 0.30, "social_sharing": 0.60, "time_pressure": 0.75, "domain_interest": 0.95, "financial_cushion": 0.50, "language_barrier": 0.10},
        ),
        Archetype(
            id="frontline_communicator",
            label="Vernacular Health Relayer",
            description="Translates complex public health guidelines into regional idioms and folklore for non-literate community members.",
            share_of_population=0.11,
            traits={"attention": 0.75, "digital_literacy": 0.45, "trust_in_source": 0.70, "skepticism": 0.45, "emotional_reactivity": 0.65, "social_sharing": 0.90, "time_pressure": 0.60, "domain_interest": 0.85, "financial_cushion": 0.25, "language_barrier": 0.20},
        ),
        Archetype(
            id="overburdened_auxiliary",
            label="Overburdened Multi-Village Auxiliary",
            description="Rides moped across 4 isolated hamlets; exhausted by documentation duties, skims WhatsApp directives on the go.",
            share_of_population=0.10,
            traits={"attention": 0.55, "digital_literacy": 0.35, "trust_in_source": 0.75, "skepticism": 0.35, "emotional_reactivity": 0.60, "social_sharing": 0.65, "time_pressure": 0.90, "domain_interest": 0.85, "financial_cushion": 0.25, "language_barrier": 0.25},
        ),
        Archetype(
            id="community_mobilizer",
            label="Self-Help Group Mobilizer",
            description="Coordinates women's self-help groups (SHG); mobilizes local families for health screening camps and sanitation drives.",
            share_of_population=0.09,
            traits={"attention": 0.80, "digital_literacy": 0.50, "trust_in_source": 0.65, "skepticism": 0.50, "emotional_reactivity": 0.55, "social_sharing": 0.85, "time_pressure": 0.55, "domain_interest": 0.85, "financial_cushion": 0.35, "language_barrier": 0.20},
        ),
        Archetype(
            id="skeptical_health_supervisor",
            label="Block Health Supervisor",
            description="Monitors vaccine cold chain integrity and epidemic outbreaks; immediately challenges unverified health claims and home cures.",
            share_of_population=0.08,
            traits={"attention": 0.90, "digital_literacy": 0.70, "trust_in_source": 0.40, "skepticism": 0.85, "emotional_reactivity": 0.20, "social_sharing": 0.50, "time_pressure": 0.60, "domain_interest": 0.90, "financial_cushion": 0.60, "language_barrier": 0.05},
        ),
        Archetype(
            id="empathetic_maternal_counselor",
            label="Maternal Care Counselor",
            description="Spends time counseling young mothers on iron tablets and newborn warmth; patient listener, builds strong emotional bonds.",
            share_of_population=0.10,
            traits={"attention": 0.80, "digital_literacy": 0.40, "trust_in_source": 0.75, "skepticism": 0.40, "emotional_reactivity": 0.70, "social_sharing": 0.70, "time_pressure": 0.50, "domain_interest": 0.90, "financial_cushion": 0.30, "language_barrier": 0.25},
        ),
        Archetype(
            id="digital_health_volunteer",
            label="Ayushman Digital Volunteer",
            description="Tech-proficient youth assisting villagers in generating ABHA health IDs and linking health insurance accounts.",
            share_of_population=0.07,
            traits={"attention": 0.85, "digital_literacy": 0.80, "trust_in_source": 0.60, "skepticism": 0.60, "emotional_reactivity": 0.35, "social_sharing": 0.75, "time_pressure": 0.45, "domain_interest": 0.80, "financial_cushion": 0.35, "language_barrier": 0.10},
        ),
        Archetype(
            id="cautious_field_vetter",
            label="Field Verification Veteran",
            description="Always verifies viral disease alerts with Medical Officer in Charge (MOIC) before alarming villagers.",
            share_of_population=0.06,
            traits={"attention": 0.85, "digital_literacy": 0.50, "trust_in_source": 0.50, "skepticism": 0.75, "emotional_reactivity": 0.25, "social_sharing": 0.45, "time_pressure": 0.50, "domain_interest": 0.85, "financial_cushion": 0.40, "language_barrier": 0.15},
        ),
    ],

    # 7. Gig Workers
    "gig_workers": [
        Archetype(
            id="fulltime_delivery_fleet",
            label="Quick-Commerce Rider",
            description="10-minute grocery and food delivery rider; faces relentless timer countdowns, glances at phone screen only while stopped at red lights.",
            share_of_population=0.15,
            traits={"attention": 0.50, "digital_literacy": 0.55, "trust_in_source": 0.55, "skepticism": 0.45, "emotional_reactivity": 0.60, "social_sharing": 0.65, "time_pressure": 0.95, "domain_interest": 0.75, "financial_cushion": 0.20, "language_barrier": 0.20},
        ),
        Archetype(
            id="cab_driver_owner",
            label="Rideshare Cab Owner-Driver",
            description="Drives for Ola/Uber 14 hours daily to pay vehicle loan; monitors fuel prices and incentive thresholds continuously.",
            share_of_population=0.13,
            traits={"attention": 0.65, "digital_literacy": 0.60, "trust_in_source": 0.50, "skepticism": 0.55, "emotional_reactivity": 0.50, "social_sharing": 0.60, "time_pressure": 0.85, "domain_interest": 0.80, "financial_cushion": 0.35, "language_barrier": 0.15},
        ),
        Archetype(
            id="migrant_daily_wager",
            label="Interstate Migrant Laborer",
            description="Works on urban construction sites; sends money home via neighborhood money transfer kiosks, relies on voice notes over text.",
            share_of_population=0.14,
            traits={"attention": 0.40, "digital_literacy": 0.20, "trust_in_source": 0.75, "skepticism": 0.30, "emotional_reactivity": 0.75, "social_sharing": 0.40, "time_pressure": 0.70, "domain_interest": 0.70, "financial_cushion": 0.10, "language_barrier": 0.65},
        ),
        Archetype(
            id="student_part_timer",
            label="Evening Part-Time Student Rider",
            description="Delivers food orders after college classes to cover rent and tuition; tech-literate, sensitive to payment bonuses.",
            share_of_population=0.10,
            traits={"attention": 0.75, "digital_literacy": 0.80, "trust_in_source": 0.35, "skepticism": 0.60, "emotional_reactivity": 0.50, "social_sharing": 0.70, "time_pressure": 0.80, "domain_interest": 0.75, "financial_cushion": 0.30, "language_barrier": 0.08},
        ),
        Archetype(
            id="exhausted_night_rider",
            label="Night Shift Delivery Courier",
            description="Rides 10 PM to 6 AM for late-night medicine and food orders; fatigued, operates phone on autopilot, low scrutiny of links.",
            share_of_population=0.11,
            traits={"attention": 0.35, "digital_literacy": 0.30, "trust_in_source": 0.65, "skepticism": 0.35, "emotional_reactivity": 0.65, "social_sharing": 0.45, "time_pressure": 0.85, "domain_interest": 0.65, "financial_cushion": 0.15, "language_barrier": 0.30},
        ),
        Archetype(
            id="union_whatsapp_repper",
            label="Gig Union WhatsApp Rep",
            description="Member of gig workers' welfare union; immediately warns delivery groups about platform pay cuts and fake deposit scams.",
            share_of_population=0.08,
            traits={"attention": 0.85, "digital_literacy": 0.65, "trust_in_source": 0.30, "skepticism": 0.75, "emotional_reactivity": 0.65, "social_sharing": 0.90, "time_pressure": 0.70, "domain_interest": 0.90, "financial_cushion": 0.25, "language_barrier": 0.10},
        ),
        Archetype(
            id="incentive_chaser",
            label="Weekend Incentive Chaser",
            description="Hyper-focused on completing 25 deliveries before midnight to hit Rs 800 bonus; quick to click on potential reward promos.",
            share_of_population=0.10,
            traits={"attention": 0.70, "digital_literacy": 0.55, "trust_in_source": 0.55, "skepticism": 0.35, "emotional_reactivity": 0.75, "social_sharing": 0.55, "time_pressure": 0.90, "domain_interest": 0.85, "financial_cushion": 0.20, "language_barrier": 0.20},
        ),
        Archetype(
            id="cautious_fleet_elder",
            label="Senior Auto Driver",
            description="52-year-old auto rickshaw driver; accepts only direct UPI or cash, cynical about online investment schemes and lottery texts.",
            share_of_population=0.07,
            traits={"attention": 0.60, "digital_literacy": 0.40, "trust_in_source": 0.40, "skepticism": 0.70, "emotional_reactivity": 0.35, "social_sharing": 0.40, "time_pressure": 0.60, "domain_interest": 0.65, "financial_cushion": 0.30, "language_barrier": 0.35},
        ),
        Archetype(
            id="app_switching_courier",
            label="Multi-App Juggler",
            description="Runs Zomato, Swiggy, and Porter simultaneously; filters out promotional spam immediately unless tied to current batch order.",
            share_of_population=0.06,
            traits={"attention": 0.65, "digital_literacy": 0.70, "trust_in_source": 0.35, "skepticism": 0.65, "emotional_reactivity": 0.35, "social_sharing": 0.50, "time_pressure": 0.80, "domain_interest": 0.70, "financial_cushion": 0.30, "language_barrier": 0.15},
        ),
        Archetype(
            id="debt_burdened_driver",
            label="EV Loan Burdened Rider",
            description="Struggles with weekly battery swapping fees and vehicle EMI; high desperation, susceptible to instant easy-loan messages.",
            share_of_population=0.06,
            traits={"attention": 0.60, "digital_literacy": 0.35, "trust_in_source": 0.65, "skepticism": 0.30, "emotional_reactivity": 0.85, "social_sharing": 0.40, "time_pressure": 0.80, "domain_interest": 0.85, "financial_cushion": 0.10, "language_barrier": 0.30},
        ),
    ],

    # 8. Urban Professionals
    "urban_professionals": [
        Archetype(
            id="corporate_middle_manager",
            label="Corporate Team Lead",
            description="Manages teams in tech/consulting; inundated with Slack and Outlook, skims mobile personal messages during quick coffee breaks.",
            share_of_population=0.14,
            traits={"attention": 0.55, "digital_literacy": 0.80, "trust_in_source": 0.45, "skepticism": 0.55, "emotional_reactivity": 0.40, "social_sharing": 0.45, "time_pressure": 0.85, "domain_interest": 0.70, "financial_cushion": 0.75, "language_barrier": 0.02},
        ),
        Archetype(
            id="fintech_early_adopter",
            label="Fintech Power User",
            description="Trades equities, experiments with credit card rewards and crypto; high digital literacy, actively engages with product launches.",
            share_of_population=0.12,
            traits={"attention": 0.80, "digital_literacy": 0.90, "trust_in_source": 0.35, "skepticism": 0.65, "emotional_reactivity": 0.45, "social_sharing": 0.65, "time_pressure": 0.55, "domain_interest": 0.90, "financial_cushion": 0.80, "language_barrier": 0.02},
        ),
        Archetype(
            id="cynical_security_engineer",
            label="Infosec Skeptic",
            description="Security professional; inspects headers, flags suspicious links, immediately reports simulated lures to IT teams.",
            share_of_population=0.09,
            traits={"attention": 0.95, "digital_literacy": 0.98, "trust_in_source": 0.10, "skepticism": 0.95, "emotional_reactivity": 0.10, "social_sharing": 0.30, "time_pressure": 0.40, "domain_interest": 0.85, "financial_cushion": 0.85, "language_barrier": 0.01},
        ),
        Archetype(
            id="burnt_out_consultant",
            label="Burnt-Out Consultant",
            description="Constant travel and client crises; sleep-deprived and fatigued, makes hurried clicks on phone without verifying details.",
            share_of_population=0.11,
            traits={"attention": 0.40, "digital_literacy": 0.38, "trust_in_source": 0.50, "skepticism": 0.45, "emotional_reactivity": 0.65, "social_sharing": 0.35, "time_pressure": 0.95, "domain_interest": 0.65, "financial_cushion": 0.75, "language_barrier": 0.05},
        ),
        Archetype(
            id="aspirational_wealth_builder",
            label="Aspirational Wealth Investor",
            description="Looking to buy second apartment or high-yield bonds; open to premium wealth advisory and exclusive investment invites.",
            share_of_population=0.10,
            traits={"attention": 0.75, "digital_literacy": 0.75, "trust_in_source": 0.50, "skepticism": 0.55, "emotional_reactivity": 0.55, "social_sharing": 0.45, "time_pressure": 0.60, "domain_interest": 0.85, "financial_cushion": 0.85, "language_barrier": 0.02},
        ),
        Archetype(
            id="socially_conscious_creative",
            label="Socially Conscious Creative",
            description="Works in design/media; actively amplifies civic petitions, environmental issues, and independent creator campaigns.",
            share_of_population=0.10,
            traits={"attention": 0.80, "digital_literacy": 0.80, "trust_in_source": 0.30, "skepticism": 0.60, "emotional_reactivity": 0.80, "social_sharing": 0.85, "time_pressure": 0.50, "domain_interest": 0.80, "financial_cushion": 0.55, "language_barrier": 0.02},
        ),
        Archetype(
            id="pragmatic_accountant",
            label="Chartered Accountant Auditor",
            description="Audits balance sheets; checks official gazette notifications, questions tax evasion lures and exaggerated claims.",
            share_of_population=0.09,
            traits={"attention": 0.85, "digital_literacy": 0.70, "trust_in_source": 0.50, "skepticism": 0.80, "emotional_reactivity": 0.20, "social_sharing": 0.30, "time_pressure": 0.65, "domain_interest": 0.80, "financial_cushion": 0.75, "language_barrier": 0.02},
        ),
        Archetype(
            id="work_from_home_coder",
            label="Remote Software Engineer",
            description="Spends all day on terminal and Discord; ignores phone calls and generic SMS, highly responsive to developer tooling.",
            share_of_population=0.09,
            traits={"attention": 0.80, "digital_literacy": 0.90, "trust_in_source": 0.25, "skepticism": 0.75, "emotional_reactivity": 0.25, "social_sharing": 0.40, "time_pressure": 0.50, "domain_interest": 0.85, "financial_cushion": 0.80, "language_barrier": 0.02},
        ),
        Archetype(
            id="busy_hr_executive",
            label="Corporate HR Specialist",
            description="Coordinates corporate employee benefits; sensitive to employee healthcare policies and official labor circulars.",
            share_of_population=0.08,
            traits={"attention": 0.70, "digital_literacy": 0.70, "trust_in_source": 0.60, "skepticism": 0.55, "emotional_reactivity": 0.45, "social_sharing": 0.60, "time_pressure": 0.70, "domain_interest": 0.75, "financial_cushion": 0.65, "language_barrier": 0.02},
        ),
        Archetype(
            id="non_tech_admin",
            label="Administrative Staff Officer",
            description="Manages office facilities and vendor paperwork; uncomfortable with cybersecurity warnings, prone to clicking vendor bill links.",
            share_of_population=0.08,
            traits={"attention": 0.55, "digital_literacy": 0.35, "trust_in_source": 0.70, "skepticism": 0.35, "emotional_reactivity": 0.60, "social_sharing": 0.40, "time_pressure": 0.70, "domain_interest": 0.60, "financial_cushion": 0.45, "language_barrier": 0.15},
        ),
    ],

    # 9. Homemakers
    "homemakers": [
        Archetype(
            id="household_cfo",
            label="Household Budget CFO",
            description="Manages monthly domestic budget with precision; hunts for grocery discounts, utility bill cashbacks, and kitchen deals.",
            share_of_population=0.14,
            traits={"attention": 0.75, "digital_literacy": 0.50, "trust_in_source": 0.60, "skepticism": 0.55, "emotional_reactivity": 0.45, "social_sharing": 0.60, "time_pressure": 0.50, "domain_interest": 0.85, "financial_cushion": 0.50, "language_barrier": 0.15},
        ),
        Archetype(
            id="family_whatsapp_central",
            label="Extended Family Hub",
            description="The central node linking relatives across cities; regularly shares devotional cards, festival greetings, and family news.",
            share_of_population=0.13,
            traits={"attention": 0.70, "digital_literacy": 0.25, "trust_in_source": 0.80, "skepticism": 0.25, "emotional_reactivity": 0.80, "social_sharing": 0.95, "time_pressure": 0.35, "domain_interest": 0.75, "financial_cushion": 0.45, "language_barrier": 0.25},
        ),
        Archetype(
            id="health_remedy_enthusiast",
            label="Ayurvedic Wellness Enthusiast",
            description="Follows herbal nutrition tips and home cures; forwards remedies for seasonal flu, joint pain, and dengue fever.",
            share_of_population=0.12,
            traits={"attention": 0.75, "digital_literacy": 0.30, "trust_in_source": 0.70, "skepticism": 0.30, "emotional_reactivity": 0.75, "social_sharing": 0.80, "time_pressure": 0.40, "domain_interest": 0.90, "financial_cushion": 0.50, "language_barrier": 0.20},
        ),
        Archetype(
            id="society_kitty_member",
            label="Neighborhood Social Organizer",
            description="Active in residential society kitty group and cultural committees; quick to relay news about community safety or events.",
            share_of_population=0.10,
            traits={"attention": 0.80, "digital_literacy": 0.55, "trust_in_source": 0.60, "skepticism": 0.45, "emotional_reactivity": 0.65, "social_sharing": 0.85, "time_pressure": 0.45, "domain_interest": 0.80, "financial_cushion": 0.60, "language_barrier": 0.10},
        ),
        Archetype(
            id="anxious_board_exam_mom",
            label="Exam Anxious Parent",
            description="Stressed about child's 10th/12th board exams; receptive to career guidance notices, coaching offers, and education news.",
            share_of_population=0.11,
            traits={"attention": 0.80, "digital_literacy": 0.45, "trust_in_source": 0.65, "skepticism": 0.35, "emotional_reactivity": 0.85, "social_sharing": 0.70, "time_pressure": 0.75, "domain_interest": 0.90, "financial_cushion": 0.50, "language_barrier": 0.15},
        ),
        Archetype(
            id="resilient_artisan_homemaker",
            label="Home Baker / Artisan Entrepreneur",
            description="Runs micro home enterprise selling snacks or crafts on WhatsApp status; understands UPI payments, careful with money.",
            share_of_population=0.09,
            traits={"attention": 0.80, "digital_literacy": 0.65, "trust_in_source": 0.45, "skepticism": 0.60, "emotional_reactivity": 0.40, "social_sharing": 0.75, "time_pressure": 0.60, "domain_interest": 0.85, "financial_cushion": 0.40, "language_barrier": 0.15},
        ),
        Archetype(
            id="cautious_senior_matron",
            label="Traditional Conservative Matron",
            description="Cautious of digital banking; insists on cash or having children handle online links, distrusts unverified numbers.",
            share_of_population=0.10,
            traits={"attention": 0.60, "digital_literacy": 0.20, "trust_in_source": 0.45, "skepticism": 0.75, "emotional_reactivity": 0.45, "social_sharing": 0.40, "time_pressure": 0.25, "domain_interest": 0.60, "financial_cushion": 0.45, "language_barrier": 0.40},
        ),
        Archetype(
            id="discount_coupon_hunter",
            label="Flash Deal & Cashback Seeker",
            description="Enthusiastically tracks festive flash sales and grocery cashback codes on Zepto/Swiggy Instamart; clicks promo links.",
            share_of_population=0.08,
            traits={"attention": 0.75, "digital_literacy": 0.60, "trust_in_source": 0.55, "skepticism": 0.35, "emotional_reactivity": 0.65, "social_sharing": 0.65, "time_pressure": 0.55, "domain_interest": 0.85, "financial_cushion": 0.45, "language_barrier": 0.10},
        ),
        Archetype(
            id="local_civic_advocate",
            label="Civic & Ward Volunteer",
            description="Active in ward waste segregation and water supply groups; circulates official municipal reminders and utility notices.",
            share_of_population=0.07,
            traits={"attention": 0.85, "digital_literacy": 0.55, "trust_in_source": 0.60, "skepticism": 0.65, "emotional_reactivity": 0.50, "social_sharing": 0.80, "time_pressure": 0.45, "domain_interest": 0.85, "financial_cushion": 0.55, "language_barrier": 0.10},
        ),
        Archetype(
            id="quiet_digital_observer",
            label="Quiet Digital Observer",
            description="Follows family and community groups silently without posting; cautious and skeptical of chain forwards and miraculous claims.",
            share_of_population=0.06,
            traits={"attention": 0.65, "digital_literacy": 0.50, "trust_in_source": 0.35, "skepticism": 0.70, "emotional_reactivity": 0.25, "social_sharing": 0.15, "time_pressure": 0.30, "domain_interest": 0.55, "financial_cushion": 0.50, "language_barrier": 0.15},
        ),
    ],
}


# --------------------------------------------------------------------------
# Validation & Normalization Helpers
# --------------------------------------------------------------------------

def validate_and_normalize_archetypes(
    archetype_dicts: List[Dict[str, Any]]
) -> Optional[List[Archetype]]:
    """Validate, clamp traits, check diversity rules, and normalize shares.

    Rules:
    - Must have at least 2 skeptical segments (skepticism > 0.6).
    - Must have at least 2 vulnerable segments (digital_literacy < 0.4).
    - Traits must be clamped between 0.0 and 1.0.
    - Population shares must be normalized to sum to 1.0.
    """
    if not isinstance(archetype_dicts, list) or len(archetype_dicts) < 4:
        return None

    parsed: List[Archetype] = []
    skeptical_count = 0
    vulnerable_count = 0

    for idx, item in enumerate(archetype_dicts):
        if not isinstance(item, dict):
            return None

        raw_id = str(item.get("id", f"segment_{idx+1}")).strip().lower().replace(" ", "_")
        label = str(item.get("label", "")).strip() or raw_id.replace("_", " ").title()
        description = str(item.get("description", "")).strip() or label
        raw_share = float(item.get("share_of_population", 1.0 / len(archetype_dicts)))

        raw_traits = item.get("traits", {})
        if not isinstance(raw_traits, dict):
            raw_traits = {}

        # Clamp all universal traits to [0.0, 1.0]
        clamped_traits: Dict[str, float] = {}
        for trait_name in TRAITS:
            val = float(raw_traits.get(trait_name, 0.5))
            clamped_traits[trait_name] = max(0.0, min(1.0, round(val, 3)))

        if clamped_traits["skepticism"] > 0.6:
            skeptical_count += 1
        if clamped_traits["digital_literacy"] < 0.4:
            vulnerable_count += 1

        parsed.append(
            Archetype(
                id=raw_id,
                label=label,
                description=description,
                share_of_population=max(0.001, raw_share),
                traits=clamped_traits,
            )
        )

    # Diversity check: at least 2 skeptical & 2 vulnerable segments
    if skeptical_count < 2 or vulnerable_count < 2:
        logger.warning(
            "Archetype diversity check failed: skeptics=%d, vulnerable=%d",
            skeptical_count,
            vulnerable_count,
        )
        return None

    # Renormalize shares to sum to 1.0
    total_share = sum(a.share_of_population for a in parsed)
    if total_share <= 0:
        total_share = 1.0

    for a in parsed:
        a.share_of_population = round(a.share_of_population / total_share, 4)

    # Ensure exact 1.0 sum
    diff = round(1.0 - sum(a.share_of_population for a in parsed), 4)
    parsed[-1].share_of_population = round(parsed[-1].share_of_population + diff, 4)

    return parsed


def find_closest_preset(desc: str) -> str:
    """Find the best-matching PRESET key for an audience description."""
    desc_lower = desc.lower()
    scores: Dict[str, int] = {k: 0 for k in PRESETS}

    keywords_map: Dict[str, List[str]] = {
        "urban_seniors": ["senior", "elder", "retire", "pension", "old", "grandparent", "aging", "aged", "60+", "geriatric"],
        "college_students": ["student", "college", "campus", "university", "youth", "teen", "grad", "hostel", "fresher", "study", "undergrad", "learner"],
        "rural_farmers": ["farmer", "rural", "village", "agri", "crop", "kisan", "mandi", "field", "harvest", "panchayat", "grower", "cultivat", "cotton", "wheat", "landholder"],
        "working_parents": ["parent", "mother", "father", "child", "school", "family", "mom", "dad", "kid", "toddler", "baby", "children"],
        "small_shopkeepers": ["shopkeeper", "retail", "merchant", "kirana", "trader", "store", "vendor", "stall", "market", "khata", "dukaan", "bazaar", "shop"],
        "community_health_workers": ["health", "asha", "anganwadi", "nurse", "clinic", "medical", "doctor", "frontline", "phc", "hospital", "healthcare", "pharma"],
        "gig_workers": ["gig", "delivery", "driver", "rider", "courier", "daily-wage", "wage", "zomato", "swiggy", "uber", "ola", "laborer", "worker", "contract"],
        "urban_professionals": ["professional", "corporate", "office", "tech", "manager", "software", "executive", "engineer", "desk", "coder", "developer", "consultant", "white collar"],
        "homemakers": ["homemaker", "housewife", "home", "domestic", "neighborhood", "society", "colony", "family group", "kitchen", "stay-at-home"],
    }

    for preset_id, words in keywords_map.items():
        for w in words:
            if w in desc_lower:
                scores[preset_id] += 3

    for preset_id in PRESETS:
        for part in preset_id.split("_"):
            if part in desc_lower:
                scores[preset_id] += 1

    best_preset = max(scores, key=lambda k: scores[k])
    if scores[best_preset] == 0:
        return "urban_seniors"
    return best_preset


# --------------------------------------------------------------------------
# LLM Generation with Retries, Validation, and Caching
# --------------------------------------------------------------------------

def generate_archetypes(audience_description: str, n: int = 10) -> List[Archetype]:
    """Generate or retrieve cached custom audience segments using local LLM.

    Args:
        audience_description: Custom user-provided audience description.
        n: Desired number of distinct segments (default 10).

    Returns:
        List of validated Archetype objects with shares summing to 1.0.
    """
    clean_desc = audience_description.strip()
    if not clean_desc:
        clean_desc = "General urban Indian population"

    cache_id = hash_key(f"{clean_desc}:{n}")
    cached_data = get_cache("archetypes", cache_id)
    if cached_data and isinstance(cached_data, list):
        try:
            return [Archetype(**item) for item in cached_data]
        except Exception:
            pass

    prompt = (
        f"You design realistic audience segments for a communication research simulation. "
        f"Audience: {clean_desc}. Return {n} DISTINCT segments covering age, income, "
        f"tech comfort, language and attitudes. Include minorities (skeptics, power users, vulnerable people). "
        f"Output JSON only: {{\"archetypes\":[{{\"id\":\"...\",\"label\":\"...\",\"description\":\"...\","
        f"\"share_of_population\":0.1,\"traits\":{{\"attention\":0.5,\"digital_literacy\":0.5,"
        f"\"trust_in_source\":0.5,\"skepticism\":0.5,\"emotional_reactivity\":0.5,\"social_sharing\":0.5,"
        f"\"time_pressure\":0.5,\"domain_interest\":0.5,\"financial_cushion\":0.5,\"language_barrier\":0.5}}]}}}}"
    )

    # Attempt up to 2 retries (3 total attempts)
    for attempt in range(3):
        try:
            data = query_ollama(prompt=prompt, json_schema=ARCHETYPES_JSON_SCHEMA, timeout=40.0)
            raw_archetypes = data.get("archetypes", [])
            valid_result = validate_and_normalize_archetypes(raw_archetypes)
            if valid_result:
                # Save to cache
                set_cache("archetypes", cache_id, [a.model_dump() for a in valid_result])
                return valid_result
            logger.warning("Attempt %d: Archetypes validation failed, retrying...", attempt + 1)
        except Exception as exc:
            logger.warning("Attempt %d LLM generation error: %s", attempt + 1, exc)

    # Fallback to closest preset if all attempts fail
    closest_preset_key = find_closest_preset(clean_desc)
    logger.info("Falling back to preset '%s' for audience '%s'", closest_preset_key, clean_desc)
    preset_copy = [a.model_copy() for a in PRESETS[closest_preset_key]]
    set_cache("archetypes", cache_id, [a.model_dump() for a in preset_copy])
    return preset_copy[:n] if len(preset_copy) >= n else preset_copy
