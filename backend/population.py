"""
population.py  -  Persona
=========================
Generates synthetic Indian populations sampled from behavioral archetype distributions,
adding Gaussian trait jitter, plausible names by language/region, and realistic demographic context.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional
import numpy as np

from backend.domain_packs import TRAITS
from backend.models import Archetype, Person

# --------------------------------------------------------------------------
# Authentic Indian Name Pools categorized by language and cultural region
# --------------------------------------------------------------------------
NAMES_BY_LANGUAGE: Dict[str, Dict[str, List[str]]] = {
    "hi": {  # Hindi / North
        "first_male": [
            "Rajesh", "Amit", "Ramesh", "Suresh", "Manoj", "Vikram", "Sanjay",
            "Deepak", "Rahul", "Ajay", "Rohit", "Anil", "Sunil", "Pankaj", "Vikas",
            "Alok", "Dinesh", "Gaurav", "Harish", "Kapil"
        ],
        "first_female": [
            "Sunita", "Pooja", "Rekha", "Geeta", "Anita", "Meena", "Kavita",
            "Seema", "Priya", "Neha", "Ritu", "Anjali", "Shashi", "Mamta", "Sangeeta",
            "Vandana", "Sarita", "Urmila", "Archana", "Champa"
        ],
        "last": [
            "Sharma", "Verma", "Gupta", "Singh", "Yadav", "Mishra", "Tiwari",
            "Agarwal", "Chauhan", "Dubey", "Pandey", "Kumar", "Srivastava", "Shukla"
        ],
    },
    "mr": {  # Marathi / Western
        "first_male": [
            "Sachin", "Prashant", "Nitin", "Nilesh", "Milind", "Ganesh", "Amol",
            "Vinod", "Mahesh", "Suhas", "Santosh", "Tushar", "Dattatray", "Hemant"
        ],
        "first_female": [
            "Anjali", "Snehal", "Swati", "Pallavi", "Madhuri", "Varsha", "Vaishali",
            "Suvarna", "Manisha", "Aparna", "Deepali", "Pratibha", "Rohini"
        ],
        "last": [
            "Patil", "Deshmukh", "Kulkarni", "Shinde", "Joshi", "More", "Gaikwad",
            "Pawar", "Chavan", "Kale", "Bhosale", "Kadam"
        ],
    },
    "ta": {  # Tamil / Southern
        "first_male": [
            "Karthik", "Senthil", "Murugan", "Vijay", "Saravanan", "Vignesh",
            "Balaji", "Praveen", "Suresh", "Ramesh", "Muthu", "Aravind", "Sundar"
        ],
        "first_female": [
            "Priya", "Lakshmi", "Meenakshi", "Revathi", "Malathi", "Deepa",
            "Uma", "Kavitha", "Bhuvaneshwari", "Divya", "Padma", "Gayathri"
        ],
        "last": [
            "Iyer", "Sundaram", "Subramanian", "Natarajan", "Balakrishnan",
            "Raman", "Swaminathan", "Narayanan", "Krishnan", "Mudaliar", "Ganesan"
        ],
    },
    "te": {  # Telugu / Southern
        "first_male": [
            "Srinivas", "Venkat", "Prasad", "Naresh", "Ravi", "Kalyan",
            "Chaitanya", "Vamsi", "Phani", "Sridhar", "Bhaskar", "Subba Rao"
        ],
        "first_female": [
            "Padmaja", "Sandhya", "Anuradha", "Swapna", "Sireesha", "Haritha",
            "Bhavani", "Lalitha", "Sujatha", "Roopa", "Kavitha", "Sailaja"
        ],
        "last": [
            "Reddy", "Rao", "Chowdary", "Raju", "Murthy", "Varma", "Goud", "Naidu"
        ],
    },
    "bn": {  # Bengali / Eastern
        "first_male": [
            "Sourav", "Anirban", "Subhash", "Debabrata", "Prosenjit", "Indranil",
            "Abhijit", "Ayan", "Tanmoy", "Bikram", "Partha", "Dipankar"
        ],
        "first_female": [
            "Debolina", "Moushumi", "Sharmila", "Rupa", "Tanusree", "Swagata",
            "Payel", "Soma", "Madhumita", "Kakali", "Nandita"
        ],
        "last": [
            "Chatterjee", "Mukherjee", "Banerjee", "Ghosh", "Sen", "Das",
            "Roy", "Bose", "Dutta", "Chakraborty", "Bhattacharya"
        ],
    },
    "kn": {  # Kannada / Southern
        "first_male": [
            "Manjunath", "Chetan", "Praveen", "Basavaraj", "Raghavendra",
            "Anand", "Shankar", "Pradeep", "Mahesh", "Siddaramaiah"
        ],
        "first_female": [
            "Roopa", "Soumya", "Divya", "Shweta", "Poornima", "Girija",
            "Vidya", "Kavya", "Renuka", "Sharada"
        ],
        "last": [
            "Gowda", "Hegde", "Bhat", "Shetty", "Rao", "Patil", "Kamat", "Naik"
        ],
    },
    "en": {  # English / Urban Pan-Indian
        "first_male": [
            "Rohan", "Kabir", "Arjun", "Dev", "Aditya", "Neil", "Aryan",
            "Karan", "Reyansh", "Siddharth", "Zorawar", "Armaan"
        ],
        "first_female": [
            "Ananya", "Rhea", "Tara", "Maya", "Zoya", "Aisha", "Natasha",
            "Ishani", "Alia", "Meera", "Diya", "Sana"
        ],
        "last": [
            "Mehta", "Kapoor", "Nair", "D'Souza", "Khan", "Merchant",
            "Roy", "Saxena", "Menon", "Malhotra", "Fernandes"
        ],
    },
}


def _get_archetype_demographics(
    arch_id: str,
    arch_desc: str,
    py_rng: random.Random
) -> tuple[int, str, str]:
    """Determine plausible age range, area, and primary language from archetype context.

    Returns:
        (age, area, language)
    """
    id_lower = arch_id.lower()
    desc_lower = arch_desc.lower()

    # 1. Age range determination
    if any(k in id_lower for k in ["senior", "pensioner", "elder", "widow", "grandparent", "matriarch"]):
        min_age, max_age = 58, 78
    elif any(k in id_lower for k in ["student", "crammer", "fresher", "intern", "undergrad"]):
        min_age, max_age = 18, 25
    elif any(k in id_lower for k in ["part_time", "youth", "influencer", "streamer", "successor"]):
        min_age, max_age = 21, 30
    elif any(k in id_lower for k in ["parent", "mother", "father", "carer"]):
        min_age, max_age = 29, 49
    elif any(k in id_lower for k in ["shopkeeper", "trader", "merchant", "dealer", "kirana"]):
        min_age, max_age = 28, 62
    elif any(k in id_lower for k in ["health", "asha", "anganwadi", "nurse"]):
        min_age, max_age = 25, 54
    elif any(k in id_lower for k in ["gig", "rider", "driver", "courier", "wager"]):
        min_age, max_age = 21, 46
    elif any(k in id_lower for k in ["professional", "manager", "engineer", "coder", "consultant"]):
        min_age, max_age = 25, 52
    elif any(k in id_lower for k in ["homemaker", "housewife"]):
        min_age, max_age = 26, 58
    elif any(k in id_lower for k in ["farmer", "ryot", "smallholder", "grower"]):
        min_age, max_age = 25, 65
    else:
        min_age, max_age = 22, 60

    age = py_rng.randint(min_age, max_age)

    # 2. Area determination
    if any(k in id_lower for k in ["farmer", "rural", "village", "ryot", "smallholder", "sharecropper"]):
        area = "rural"
    elif any(k in id_lower for k in ["asha", "anganwadi", "kiosk", "stall"]):
        area = py_rng.choices(["rural", "suburban"], weights=[0.6, 0.4])[0]
    elif any(k in id_lower for k in ["shopkeeper", "kirana", "trader", "mechanic"]):
        area = py_rng.choices(["suburban", "metro", "rural"], weights=[0.5, 0.35, 0.15])[0]
    elif any(k in id_lower for k in ["professional", "coder", "consultant", "fintech", "manager"]):
        area = "metro"
    elif any(k in id_lower for k in ["gig", "rider", "driver", "courier"]):
        area = py_rng.choices(["metro", "suburban"], weights=[0.8, 0.2])[0]
    elif any(k in id_lower for k in ["student", "hostel", "fresher"]):
        area = py_rng.choices(["metro", "suburban"], weights=[0.75, 0.25])[0]
    elif any(k in id_lower for k in ["senior", "pensioner"]):
        area = py_rng.choices(["metro", "suburban"], weights=[0.65, 0.35])[0]
    else:
        area = py_rng.choices(["metro", "suburban", "rural"], weights=[0.5, 0.35, 0.15])[0]

    # 3. Language selection based on area and demographic
    if area == "rural":
        lang = py_rng.choices(["hi", "mr", "te", "ta", "bn", "kn"], weights=[0.35, 0.15, 0.15, 0.15, 0.10, 0.10])[0]
    elif any(k in id_lower for k in ["professional", "coder", "influencer", "fintech"]):
        lang = py_rng.choices(["en", "hi"], weights=[0.7, 0.3])[0]
    elif area == "metro":
        lang = py_rng.choices(["en", "hi", "mr", "ta", "te", "bn", "kn"], weights=[0.35, 0.30, 0.10, 0.08, 0.07, 0.05, 0.05])[0]
    else:  # suburban
        lang = py_rng.choices(["hi", "mr", "ta", "te", "bn", "kn", "en"], weights=[0.40, 0.15, 0.12, 0.12, 0.08, 0.08, 0.05])[0]

    return age, area, lang


def generate_population(
    archetypes: List[Archetype],
    n: int = 120,
    seed: int = 42
) -> List[Person]:
    """Generate a synthetic population of Person objects with realistic demographics and jittered traits.

    Args:
        archetypes: List of Archetype models defining the baseline population segments.
        n: Number of synthetic people to generate (default 120).
        seed: Deterministic random seed for reproducibility.

    Returns:
        List of generated Person objects.
    """
    if not archetypes:
        raise ValueError("Cannot generate population from empty archetypes list")
    if n <= 0:
        return []

    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # Calculate sampling distribution based on archetype population shares
    total_share = sum(a.share_of_population for a in archetypes)
    if total_share <= 0:
        shares = [1.0 / len(archetypes)] * len(archetypes)
    else:
        shares = [a.share_of_population / total_share for a in archetypes]

    # Guarantee at least 1 person per archetype when n >= len(archetypes)
    assigned_indices: List[int] = []
    if n >= len(archetypes):
        assigned_indices.extend(range(len(archetypes)))
        remaining = n - len(archetypes)
        if remaining > 0:
            extra = np_rng.choice(len(archetypes), size=remaining, p=shares)
            assigned_indices.extend(extra.tolist())
    else:
        chosen = np_rng.choice(len(archetypes), size=n, p=shares)
        assigned_indices.extend(chosen.tolist())

    # Shuffle assignment order deterministically
    py_rng.shuffle(assigned_indices)

    people: List[Person] = []
    for i, arch_idx in enumerate(assigned_indices):
        arch = archetypes[arch_idx]
        pid = f"p_{i+1:03d}"

        # Determine demographic profile
        age, area, lang = _get_archetype_demographics(arch.id, arch.description, py_rng)

        # Generate plausible Indian name
        lang_pool = NAMES_BY_LANGUAGE.get(lang, NAMES_BY_LANGUAGE["hi"])
        gender = py_rng.choice(["male", "female"])
        first_names = lang_pool["first_male"] if gender == "male" else lang_pool["first_female"]
        first_name = py_rng.choice(first_names)
        last_name = py_rng.choice(lang_pool["last"])
        full_name = f"{first_name} {last_name}"

        # Trait means + Gaussian jitter (sd=0.08), strictly clipped to [0.0, 1.0]
        person_traits: Dict[str, float] = {}
        for trait_name in TRAITS:
            mean_trait = arch.traits.get(trait_name, 0.5)
            jitter = float(np_rng.normal(loc=0.0, scale=0.08))
            clamped_val = max(0.0, min(1.0, round(mean_trait + jitter, 4)))
            person_traits[trait_name] = clamped_val

        person = Person(
            id=pid,
            name=full_name,
            archetype_id=arch.id,
            age=age,
            language=lang,
            area=area,
            traits=person_traits,
        )
        people.append(person)

    return people
