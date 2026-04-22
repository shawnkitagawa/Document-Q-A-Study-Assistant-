# deterministic mapping no LLM no Rag 
from collections import deque
import re
import itertools
from typing import Any 

DEFAULT_VARS = {
    "primary_ingredient": "chicken",
    "intent": "impress guests",
    "context_settings": "home dinner for 4",
    "equipment_constraints": "stovetop and oven",
    "flavor_preferences": "savory with balanced acidity",
    "cuisine_preference": "Spanish",
    "time_limit_minutes": 45,
    "primary_method": "sear and roast",
    "servings": 4,
    "ingredient_blacklist": "none",
    "ingredient_whitelist": "common pantry items",
    "dietary_constraints": "none",
    "skill_level": "home cook",
}

# 1) Use a fixed list of possible decision keys 
STABLE = {
    "method_selection": [
        "How should the primary ingredient be cooked?",
        "What is the primary cook method (the main heat technique) that best expresses the ingredient and intent?",
        "Are there any secondary steps that enhance the result without adding fragility or risk?"
    ],

    "doneness_and_heat_control": [
        "How do we control doneness and avoid overcooking?",
        "What is the target doneness or internal temperature range, and what carryover do we expect during rest or finishing?",
        "Where is the highest risk of overshoot (pan too hot, sauce reduction, reheating), and what is the control plan (lower heat, basting, staging, thermometer)?"
    ],

    "texture_target": [
        "What texture are we aiming for overall?",
        "What contrast do we want in one bite (crisp/creamy, tender/crunchy, juicy/snappy), and what component provides it?",
        "How do we protect texture over time (steam sogginess, sauce soaking, resting too long), especially if it sits 5–10 minutes?"
    ],

    "sauce_architecture": [
        "Do we need a sauce, and what kind of structure should it have?",
        "What is the sauce’s job: moisture, richness, acidity lift, aromatic carrier, or heat delivery—and which matters most?",
        "What is the base structure (emulsion, reduction, beurre monté, stock gel, purée, vinaigrette), and how will it stay stable (no splitting or breaking)?"
    ],

    "umami_strategy": [
        "Where does depth and savoriness come from?",
        "What is the umami layer map (base → mid → top), and which layer is optional?",
        "What is the ‘clean finish’ plan so umami doesn’t become heavy or muddy (acid, bitterness, herbs, or restrained reduction)?"
    ],

    "acid_strategy": [
        "How and when do we introduce acidity?",
        "What acid type matches the cuisine and ingredients (citrus, vinegar, wine, yogurt, tomato, pickles), and why that one?",
        "At what moments do we add acid (marinade vs finish vs sauce) to keep it bright instead of cooked-out or harsh?"
    ],

    "salt_strategy": [
        "When and how should salting happen?",
        "Where should salt live: deep seasoning (dry brine), surface seasoning, or in the sauce—and what’s the primary lever?",
        "How do we avoid salting mistakes with reductions, stocks, soy, or cheese—what gets salted last on purpose?"
    ],

    "aromatics_timing": [
        "When should aromatics be added to avoid bitterness or dullness?",
        "Which aromatics are ‘early’ vs ‘late’, and what’s the plan?",
        "What aromas should be infused then removed to prevent bitterness or perfume overload?"
    ],

    "spice_strategy": [
        "How do we build heat without bitterness or harshness?",
        "What kind of heat do we want (warmth, sharp chili bite, tingle, peppery), and what ingredient delivers that cleanly?",
        "How do we layer spice so it tastes rounded, not raw?"
    ],

    "cuisine_alignment": [
        "How do we stay stylistically coherent?",
        "What are the non-negotiable cues of the chosen cuisine (fat type, acid type, core aromatics, garnish style), and are we honoring them?",
        "If we’re blending cuisines, what are the rules: one base cuisine + one accent cuisine, and what stays sacred?"
    ],

    "fat_strategy": [
        "What role does fat play, and how much is appropriate?",
        "Where is fat doing the work (searing medium, emulsion, finishing gloss, mouthfeel), and what happens if we reduce it 20%?",
        "What fat choice matches the story, and is its flavor intentional or neutral?"
    ],

    "plating_and_finish": [
        "How should the dish be finished and presented?",
        "What is the ‘last 30 seconds’ plan that makes it taste alive?",
        "How do we plate to preserve temperature and texture?"
    ],

    "substitution_strategy": [
        "What swaps maintain intent if something is unavailable?",
        "If we swap an ingredient, what function must be preserved before we pick the substitute?",
        "What is the best closest-by-behavior substitute for each critical component (primary ingredient, thickener, acid, crunch, herb)?"
    ],

    "failure_prevention": [
        "What steps are most likely to go wrong, and how do we avoid them?",
        "What are the top failure modes, and what are the early warning signs for each?",
        "What is the rescue plan mid-cook to correct texture or flavor?"
    ],

    "leftover_or_scaling_strategy": [
        "Can this scale or store well?",
        "Which components can be made ahead without quality loss, and what must be cooked à la minute?",
        "If doubling the recipe, what changes to keep results identical?"
    ],

    "temperature_strategy": [
        "What temperature ranges should be used at each stage and why?",
        "What is the ideal serving temperature for each component (primary ingredient, sauce, garnish), and how do we keep them there?",
        "What hot/cold contrast could improve the experience without feeling strange?"
    ],

    "time_strategy": [
        "How should the total time be allocated across steps?",
        "What is the critical path, and how do we schedule around it?",
        "What can be done in parallel so the primary ingredient finishes last and rests properly?"
    ],

    "visual_strategy": [
        "How should the dish look to reinforce appetite, balance, and intent?",
        "What is the visual focal point (primary ingredient, sauce sheen, garnish), and what is the negative space plan?",
        "Do color and shape communicate the intended mood?"
    ],

    "audience_affinity": [
        "What eater profiles is this dish naturally appealing to?",
        "Who might not like it, and can we offer an easy adjustment without ruining identity?",
        "What is the comfort hook that makes a new dish feel safe?"
    ],

    "context_settings": [
        "What does this meal need to optimize for above all else?",
        "How much complexity or risk is acceptable for this context?",
        "How important is presentation relative to practicality?"
    ],

    "intent_preservation_strategy": [
        "What is the core intent of this dish in one sentence?",
        "Which single compromise would destroy the intent fastest, and how do we prevent it?",
        "If we must simplify, what stays and what goes so the dish still reads as the same idea?"
    ],

    "flavor_pairing_logic": [
        "What is the pairing relationship: echo, contrast, or bridge?",
        "Which dominant notes must harmonize, and what is the bridge ingredient?",
        "What is the anti-pairing risk, and what prevents it?"
    ],

    "fermentation_and_aging_strategy": [
        "Would a transformed ingredient improve depth more than cooking alone?",
        "Are we using fermentation for umami, acidity, aroma, or texture?",
        "Where do we place the fermented element so it reads as integrated?"
    ],

    "browning_and_aroma_chemistry": [
        "Do we want this dish to taste browned and roasty or clean and delicate?",
        "Where do we intentionally create browning, and where must we avoid it?",
        "What gets bloomed or infused versus kept fresh to maximize aroma?"
    ],

    "signature_move": [
        "What is the one signature move that makes this dish memorable?",
        "Does the signature move reinforce the intent and cuisine cues?",
        "Can the signature move be executed consistently under real constraints?"
    ]
}


TEMPLATES = {

  "method_selection": [
    "chef decision logic: overall approach for {primary_ingredient} to express intent {intent} under equipment {equipment_constraints}",
    "dominant heat technique selection for {primary_ingredient} given intent {intent} and constraints {equipment_constraints}",
    "supporting transformations (cure/ferment/brine/smoke/age) for {primary_ingredient} that fit intent {intent} and context {context_settings}"
  ],

  "doneness_and_heat_control": [
    "doneness control plan for {primary_ingredient} using method {primary_method} under time {time_limit_minutes} minutes",
    "carryover and resting strategy for {primary_ingredient} cooked via {primary_method} given servings {servings}",
    "highest overshoot risks for {primary_ingredient} with {primary_method} under constraints {equipment_constraints} and how to control them"
  ],

  "texture_target": [
    "dominant texture goal for {primary_ingredient} given intent {intent} and audience context {context_settings}",
    "designing contrast textures around {primary_ingredient} for flavor {flavor_preferences} and intent {intent}",
    "protecting crispness/juiciness for {primary_ingredient} under plating constraints {context_settings} and time {time_limit_minutes}"
  ],

  "sauce_architecture": [
    "sauce structure choice for {primary_ingredient} cooked by {primary_method} aligned with intent {intent}",
    "stability tradeoffs: emulsion vs reduction for intent {intent} under equipment {equipment_constraints}",
    "sauce role prioritization (moisture/richness/acid lift/aroma) for intent {intent} and flavor {flavor_preferences}"
  ],

  "umami_strategy": [
    "umami layering plan for {primary_ingredient} in cuisine {cuisine_preference} aligned with intent {intent}",
    "umami intensity control for intent {intent} with flavor {flavor_preferences} to avoid muddiness",
    "clean finish plan for umami-forward {primary_ingredient} dish given acid preferences {flavor_preferences} and context {context_settings}"
  ],

  "acid_strategy": [
    "acid timing decisions for {primary_ingredient} with method {primary_method} to support intent {intent}",
    "choosing acid type to match cuisine {cuisine_preference} and constraints {ingredient_blacklist}",
    "balancing richness/umami for intent {intent} using acidity for flavor {flavor_preferences}"
  ],

  "salt_strategy": [
    "salting hierarchy for {primary_ingredient} under method {primary_method} and constraints {dietary_constraints}",
    "salt timing effects on texture for {primary_ingredient} given intent {intent} and time {time_limit_minutes}",
    "avoiding oversalting with ferments/aged elements given blacklist {ingredient_blacklist} and cuisine {cuisine_preference}"
  ],

  "aromatics_timing": [
    "aromatic timing plan for cuisine {cuisine_preference} supporting intent {intent}",
    "early vs late aromatics for {primary_ingredient} cooked with {primary_method} to avoid bitterness",
    "infuse vs fresh aromatics decision under context {context_settings} and time {time_limit_minutes}"
  ],

  "spice_strategy": [
    "heat architecture for flavor {flavor_preferences} aligned with intent {intent}",
    "spice layering method choices for cuisine {cuisine_preference} under equipment {equipment_constraints}",
    "softening aggressive heat for audience context {context_settings} while keeping flavor {flavor_preferences}"
  ],

  "cuisine_alignment": [
    "non-negotiable cues of cuisine {cuisine_preference} applied to {primary_ingredient} with intent {intent}",
    "what breaks coherence in cuisine {cuisine_preference} when constraints include {equipment_constraints}",
    "controlled fusion rules: base cuisine {cuisine_preference} + accent while preserving intent {intent}"
  ],

  "fat_strategy": [
    "fat role decisions for intent {intent} and flavor {flavor_preferences} with {primary_ingredient}",
    "reducing fat while preserving mouthfeel under context {context_settings} and dietary {dietary_constraints}",
    "fat choice coherence with cuisine {cuisine_preference} and cooking method {primary_method}"
  ],

  "plating_and_finish": [
    "last-minute finishing plan for intent {intent} and context {context_settings} for {primary_ingredient}",
    "plating to preserve texture for {primary_ingredient} given sauce structure and time {time_limit_minutes}",
    "finish decisions (herbs/zest/crunch/salt) compatible with blacklist {ingredient_blacklist} and flavor {flavor_preferences}"
  ],

  "substitution_strategy": [
    "substitution by function for {primary_ingredient} given whitelist {ingredient_whitelist} and blacklist {ingredient_blacklist}",
    "closest-by-behavior substitutes under dietary constraints {dietary_constraints} and cuisine {cuisine_preference}",
    "maintaining intent {intent} when swapping ingredients due to constraints {ingredient_blacklist}"
  ],

  "failure_prevention": [
    "top failure modes for {primary_ingredient} with method {primary_method} under constraints {equipment_constraints}",
    "early warning signs and rescue plan aligned with skill level {skill_level} and context {context_settings}",
    "mid-cook correction ladder for salt/acid/umami given flavor {flavor_preferences} and intent {intent}"
  ],

  "leftover_or_scaling_strategy": [
    "scaling dish for servings {servings} while preserving intent {intent} and method {primary_method}",
    "make-ahead vs à la minute decisions under context {context_settings} and time {time_limit_minutes}",
    "reheating/holding strategies for {primary_ingredient} given equipment constraints {equipment_constraints}"
  ],

  "temperature_strategy": [
    "ideal serving temperatures for {primary_ingredient} with method {primary_method} aligned with intent {intent}",
    "holding strategy without quality loss under context {context_settings} and equipment {equipment_constraints}",
    "temperature contrast ideas compatible with flavor {flavor_preferences} and cuisine {cuisine_preference}"
  ],

  "time_strategy": [
    "critical path schedule for {primary_ingredient} aligned with intent {intent} under time {time_limit_minutes}",
    "parallelization plan under equipment {equipment_constraints} for servings {servings}",
    "what must be à la minute vs staged for context {context_settings} and skill level {skill_level}"
  ],

  "visual_strategy": [
    "visual hierarchy and focal point for {primary_ingredient} aligned with intent {intent}",
    "color and contrast plan compatible with cuisine {cuisine_preference} and context {context_settings}",
    "plating geometry and negative space suited to servings {servings} and style intent {intent}"
  ],

  "audience_affinity": [
    "audience fit for intent {intent} and flavor {flavor_preferences} under context {context_settings}",
    "adjustment options to reduce spice/richness/acidity while preserving intent {intent}",
    "comfort anchors for romantic/low-stress context {context_settings} with {primary_ingredient}"
  ],

  "context_settings": [
    "decision tradeoffs optimizing for context {context_settings} given time {time_limit_minutes} and equipment {equipment_constraints}",
    "risk control and simplification for context {context_settings} with skill level {skill_level}",
    "presentation vs practicality for context {context_settings} given intent {intent}"
  ],

  "intent_preservation_strategy": [
    "define and preserve intent {intent} for {primary_ingredient} under constraints {equipment_constraints}",
    "simplification rules that preserve dish identity for intent {intent} with time {time_limit_minutes}",
    "what destroys intent {intent} fastest given flavor {flavor_preferences} and how to prevent it"
  ],

  "flavor_pairing_logic": [
    "pairing logic for {primary_ingredient} given flavor {flavor_preferences} and intent {intent}",
    "bridge ingredient strategies within cuisine {cuisine_preference} while respecting blacklist {ingredient_blacklist}",
    "anti-pairing risks for {primary_ingredient} under acid/salt goals from flavor {flavor_preferences}"
  ],

  "fermentation_and_aging_strategy": [
    "transformed elements (fermented/aged/cured/smoked/dried/pickled) that complement {primary_ingredient} under intent {intent}",
    "controlling funk intensity for context {context_settings} and skill level {skill_level}",
    "where to place fermented elements in the dish structure for cuisine {cuisine_preference} and constraints {ingredient_blacklist}"
  ],

  "browning_and_aroma_chemistry": [
    "browning vs clarity choice for intent {intent} with {primary_ingredient}",
    "fond development and deglazing strategy for method {primary_method} under equipment {equipment_constraints}",
    "maximize aroma without bitterness given spice preferences {flavor_preferences} and context {context_settings}"
  ],

  "signature_move": [
    "high-impact low-risk signature move for {primary_ingredient} aligned with intent {intent}",
    "signature move that fits context {context_settings} and constraints {equipment_constraints}",
    "memorable element using transformed components compatible with blacklist {ingredient_blacklist} and flavor {flavor_preferences}"
  ],
}

CANON_ORDER = [
  "context_settings",
  "intent_preservation_strategy",
  "time_strategy",
  "method_selection",
  "doneness_and_heat_control",
  "temperature_strategy",
  "texture_target",
  "fat_strategy",
  "sauce_architecture",
  "umami_strategy",
  "acid_strategy",
  "salt_strategy",
  "aromatics_timing",
  "spice_strategy",
  "browning_and_aroma_chemistry",
  "cuisine_alignment",
  "flavor_pairing_logic",
  "fermentation_and_aging_strategy",
  "signature_move",
  "substitution_strategy",
  "failure_prevention",
  "leftover_or_scaling_strategy",
  "plating_and_finish",
  "visual_strategy",
  "audience_affinity",
]


# Policy layer
# A decision key is activated only if at least one of its trigger signals is present in the normalized intent. 
def has(value):
    return value is not None and value != []


def has_any(container, values):
    if not container:
        return False
    return any(v in container for v in values)

def normalize_vars(vars: dict[str, object]) -> dict[str, list[str]]:
    normalized = {}
    for k, v in vars.items():
        if v is None:
            continue
        if isinstance(v, list):
            normalized[k] = [str(x) for x in v if x not in (None, "")]
        else:
            normalized[k] = [str(v)]
    return normalized



def required_vars(template: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", template))


def normalize_with_defaults(vars):
    merged = {**DEFAULT_VARS, **vars}  # user vars override defaults
    normalized = {}
    for k, v in merged.items():
        if isinstance(v, list):
            normalized[k] = [str(x) for x in v if x not in (None, "")]
        else:
            normalized[k] = [str(v)]
    return normalized


def fill_template(template: str, vars: dict[str, object]) -> list[str]:
    needed = required_vars(template)
    norm_vars = normalize_with_defaults(vars)  # <-- use merged defaults

    keys = sorted(needed)
    value_lists = [norm_vars[k] for k in keys]

    results = []
    for combo in itertools.product(*value_lists):
        filled = template
        for k, v in zip(keys, combo):
            filled = filled.replace("{" + k + "}", v)
        results.append(" ".join(filled.split()).strip())

    return results



def activate_keys(intent):
    active = {}

    # Guard
    if not intent["is_recipe_request"]:
        return active


    active["method_selection"] = True
    active["doneness_and_heat_control"] = True
    active["texture_target"] = True
    active["intent_preservation_strategy"] = True

    # --------------------------------------------------
    #  Time & Practicality
    # --------------------------------------------------
    active["time_strategy"] = (
        intent["time_limit_minutes"] is not None
        or has_any(intent["cooking_intent"], ["quick"])
    )

    active["temperature_strategy"] = (
        intent["primary_ingredient"] is not None
        or active["method_selection"]  # implicit dependency
    )

    active["leftover_or_scaling_strategy"] = (
        intent["servings"] is not None and intent["servings"] > 2
        or has_any(intent["context_settings"], ["meal_prep"])
    )

    # --------------------------------------------------
    # Flavor & Balance Systems
    # --------------------------------------------------
    active["umami_strategy"] = (
        has_any(intent["flavor_preferences"], ["umami"])
        or has_any(intent["cooking_intent"], ["rich", "savory"])
    )

    active["acid_strategy"] = (
        has_any(intent["flavor_preferences"], ["acidic", "bright"])
        or has_any(intent["cooking_intent"], ["clean"])
        or active["umami_strategy"]  # balance dependency
    )

    active["salt_strategy"] = (
        has_any(intent["dietary_constraints"], ["low_sodium"])
        or has_any(intent["ingredient_blacklist"], ["salt", "soy"])
        or active["umami_strategy"]
    )

    # --------------------------------------------------
    #  Sauce & Aromatics
    # --------------------------------------------------
    active["sauce_architecture"] = (
        has_any(intent["cooking_intent"], ["saucy"])
        or intent["primary_ingredient"] in ["chicken", "fish", "tofu"]
    )

    active["aromatics_timing"] = (
        intent["cuisine_preference"] is not None
        or has(intent["flavor_preferences"])
    )

    active["spice_strategy"] = (
        has_any(intent["flavor_preferences"], ["spicy"])
        or has_any(intent["cooking_intent"], ["spicy", "heat"])
    )

    # --------------------------------------------------
    #  Cuisine & Creativity (Opt-in)
    # --------------------------------------------------
    active["cuisine_alignment"] = intent["cuisine_preference"] is not None

    active["flavor_pairing_logic"] = (
        intent["cuisine_preference"] is not None
        or has_any(intent["cooking_intent"], ["creative", "experimental"])
    )

    active["fermentation_and_aging_strategy"] = (
        has_any(intent["flavor_preferences"], ["funky", "deep", "complex"])
    )

    active["browning_and_aroma_chemistry"] = (
        has_any(intent["flavor_preferences"], ["roasty", "smoky", "deep"])
    )

    active["signature_move"] = (
        has_any(intent["context_settings"], ["impressive"])
        or has_any(intent["cooking_intent"], ["creative", "restaurant_style"])
    )

    # --------------------------------------------------
    # Constraints & Safety
    # --------------------------------------------------
    active["substitution_strategy"] = (
        has(intent["ingredient_whitelist"])
        or has(intent["ingredient_blacklist"])
        or has(intent["dietary_constraints"])
    )

    active["failure_prevention"] = (
        intent["skill_level"] == "beginner"
        or has_any(intent["context_settings"], ["low_stress"])
    )

    # --------------------------------------------------
    # Scaling, Presentation, Audience
    # --------------------------------------------------
    active["plating_and_finish"] = (
        has_any(intent["context_settings"], ["impressive"])
        or has_any(intent["cooking_intent"], ["elegant"])
    )

    active["visual_strategy"] = active["plating_and_finish"]

    active["audience_affinity"] = (
        intent["servings"] is not None and intent["servings"] >= 3
        or has(intent["context_settings"])
    )

    return active


def build_queue(intent):
    active = activate_keys(intent)
    return deque([k for k in CANON_ORDER if active.get(k)])

def decision_key(queue_decision, vars):
    """
    Build steps for the Technique Plan.
    Each step now contains all stable questions and RAG queries for a given decision key,
    so that the LLM can handle them in one call instead of multiple calls per key.
    """
    steps = []

    while queue_decision:
        key = queue_decision.popleft()

        # Gather all stable questions for this key
        stable_questions = STABLE[key]  # assume STABLE[key] is a list of questions

        # Gather all RAG queries for this key, filled with vars
        rag_queries = []
        for template in TEMPLATES[key]:
            filled = fill_template(template, vars)
            if filled:  # skip empty results
                if isinstance(filled, list):
                    rag_queries.extend(filled)  # flatten
                else:
                    rag_queries.append(filled)

        # Skip steps with no RAG queries? Optional, depends on your logic
        if not rag_queries:
            rag_queries = []

        steps.append({
            "decision_key": key,
            "stable_questions": stable_questions,
            "rag_queries": rag_queries
        })

    return steps

            
        





    