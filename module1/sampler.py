"""
sampler.py
----------
Multi-Attribute Sampler: generates diverse, gender-consistent facial-attribute
vectors and human-readable descriptions from BERT-predicted confidence scores.
"""

from __future__ import annotations

import os
import re
import random
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Domain knowledge
# ---------------------------------------------------------------------------

# Exactly one member of each group may be active at a time.
EXCLUSIVE_GROUPS: List[List[str]] = [
    ["Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair", "Bald"],
    ["Straight_Hair", "Wavy_Hair", "Bangs"],          # hair-texture group
]

# Attributes shown in generated descriptions.
DESCRIPTION_ATTRS = {
    "Young", "Male",
    "Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair", "Bald",
    "No_Beard", "Goatee", "Mustache", "5_o_Clock_Shadow", "Sideburns",
    "Eyeglasses", "Wearing_Hat", "Oval_Face",
    "High_Cheekbones", "Big_Nose", "Narrow_Eyes", "Pale_Skin",
    "Rosy_Cheeks", "Bushy_Eyebrows", "Arched_Eyebrows",
    "Wearing_Earrings", "Wearing_Necklace", "Wearing_Necktie",
    "Wearing_Lipstick", "Heavy_Makeup",
    "Chubby", "Double_Chin", "Smiling",
}

# Attributes that only make sense for males.
MALE_ONLY_ATTRS = {
    "No_Beard", "Goatee", "Mustache", "5_o_Clock_Shadow",
    "Sideburns", "Wearing_Necktie",
}

# Attributes that only make sense for females.
FEMALE_ONLY_ATTRS = {
    "Wearing_Earrings", "Wearing_Necklace",
    "Wearing_Lipstick", "Heavy_Makeup", "Arched_Eyebrows",
}

# Attributes that are MUTUALLY EXCLUSIVE with being bald / having no hair.
HAIR_STYLE_ATTRS = {"Straight_Hair", "Wavy_Hair", "Bangs"}

# Keyword → (attribute, value) mapping parsed from input_text.
TEXT_POSITIVE_KEYWORDS: Dict[str, str] = {
    "young":        "Young",
    "old":          "Young",   # negated below
    "man":          "Male",
    "male":         "Male",
    "boy":          "Male",
    "glasses":      "Eyeglasses",
    "eyeglasses":   "Eyeglasses",
    "bald":         "Bald",
    "blond":        "Blond_Hair",
    "blonde":       "Blond_Hair",
    "black hair":   "Black_Hair",
    "brown hair":   "Brown_Hair",
    "gray hair":    "Gray_Hair",
    "grey hair":    "Gray_Hair",
    # "short hair" doesn't have a direct CelebA attribute; map to bangs
    # as a reasonable proxy for short hair/fringe.
    "short hair":   "Bangs",
    "wavy":         "Wavy_Hair",
    "straight":     "Straight_Hair",
    "bangs":        "Bangs",
    "fringe":       "Bangs",
    "hat":          "Wearing_Hat",
    "goatee":       "Goatee",
    "mustache":     "Mustache",
    "moustache":    "Mustache",
    "stubble":      "5_o_Clock_Shadow",
    "sideburns":    "Sideburns",
    "oval face":    "Oval_Face",
    "oval":         "Oval_Face",
    "smiling":      "Smiling",
    "smile":        "Smiling",
    "pale":         "Pale_Skin",
    "chubby":       "Chubby",
    "tie":          "Wearing_Necktie",
}

FEMALE_KEYWORDS = {"woman", "female", "girl", "lady", "she", "her"}
FACE_TRIGGER_KEYWORDS = {"face", "faced", "oval", "round", "square", "heart-shaped"}

# Maps a CelebA attribute name to the natural-language phrase we use in descriptions.
# Used to reconstruct text-stated phrases from the parsed attribute set.
ATTR_TO_NATURAL_PHRASE: Dict[str, str] = {
    "Young":            "young",
    "Male":             "man",   # gender handled separately
    "Black_Hair":       "black hair",
    "Brown_Hair":       "brown hair",
    "Blond_Hair":       "blond hair",
    "Gray_Hair":        "gray hair",
    "Bald":             "bald",
    "Straight_Hair":    "straight hair",
    "Wavy_Hair":        "wavy hair",
    "Bangs":            "bangs",
    "Eyeglasses":       "glasses",
    "Wearing_Hat":      "a hat",
    "Goatee":           "a goatee",
    "Mustache":         "a mustache",
    "5_o_Clock_Shadow": "light stubble",
    "Sideburns":        "sideburns",
    "No_Beard":         "no beard",
    "Oval_Face":        "an oval face",
    "Smiling":          "a smile",
    "Pale_Skin":        "pale skin",
    "Chubby":           "a chubby face",
    "High_Cheekbones":  "high cheekbones",
    "Big_Nose":         "a prominent nose",
    "Narrow_Eyes":      "narrow eyes",
    "Bushy_Eyebrows":   "bushy eyebrows",
    "Arched_Eyebrows":  "arched eyebrows",
    "Rosy_Cheeks":      "rosy cheeks",
    "Double_Chin":      "a double chin",
    "Wearing_Necktie":  "a necktie",
    "Wearing_Necklace": "a necklace",
    "Wearing_Earrings": "earrings",
    "Wearing_Lipstick": "makeup",
    "Heavy_Makeup":     "makeup",
}

# High-confidence threshold for auto-including predicted attributes
HIGH_CONF = 0.75
# Medium-confidence threshold — candidates for combinatorial variation
MED_CONF  = 0.45
# Low-confidence threshold — excluded from variation
LOW_CONF  = 0.20


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _word_in(text: str, word: str) -> bool:
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiAttributeSampler:
    """
    Generates multiple attribute vectors from BERT confidence scores +
    user-supplied text hints, then converts them to PyTorch tensors and
    human-readable descriptions.
    """

    def __init__(
        self,
        attributes_csv_path: str,
        low_thresh: float = LOW_CONF,
        high_thresh: float = HIGH_CONF,
    ) -> None:
        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh
        self._include_face_shape = False
        # Set during `sample_vectors()` / `_parse_input_text()`; used by
        # `generate_description()` to ensure the original input attributes
        # appear in every variant.
        self._text_stated_attrs: set = set()
        # Attributes explicitly forced to 0 due to *negative* keywords
        # (e.g. "old" => Young forced to 0, "beard" => No_Beard forced to 0).
        self._text_forced_zero_attrs: set = set()
        # Human-readable phrases directly extracted from input text.
        # Maps CelebA attr name → phrase string as it should appear in description.
        # e.g. {"Black_Hair": "black hair", "Eyeglasses": "glasses", "Young": "young"}
        self._text_stated_phrases: Dict[str, str] = {}

        df = pd.read_csv(attributes_csv_path, header=None)
        raw = [str(v).strip() for v in df.iloc[:, 0].tolist()]
        self.celeba_attrs: List[str] = [
            a for a in raw if a and a != "nan" and not a.isdigit()
        ]

    # ------------------------------------------------------------------
    # Text parsing
    # ------------------------------------------------------------------

    def _parse_input_text(
        self,
        text_matched: List[str],
        input_text: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Enrich text_matched from keyword scanning and derive forced_zero.

        Returns
        -------
        enriched   : attributes forced to 1
        forced_zero: attributes forced to 0
        """
        enriched: set = set(text_matched)
        forced_zero: set = set()
        # Track only *explicit* negative keywords for base description anchors.
        explicit_forced_zero: set = set()
        lower = input_text.lower() if input_text else ""

        self._include_face_shape = any(
            kw in lower for kw in FACE_TRIGGER_KEYWORDS
        )
        # Seed "original attributes" with the ones already matched by
        # upstream keyword scanning.
        self._text_stated_attrs = set(text_matched)
        # Build natural-phrase map from text_matched passed in upstream
        self._text_stated_phrases = {
            attr: ATTR_TO_NATURAL_PHRASE[attr]
            for attr in text_matched
            if attr in ATTR_TO_NATURAL_PHRASE
        }

        if not lower:
            self._text_forced_zero_attrs = set(explicit_forced_zero)
            return list(enriched), list(forced_zero)

        # ---- Gender ------------------------------------------------
        is_female = any(_word_in(lower, kw) for kw in FEMALE_KEYWORDS)
        is_male   = any(_word_in(lower, kw) for kw in ("man", "male", "boy"))

        if is_female and not is_male:
            forced_zero.add("Male")
            explicit_forced_zero.add("Male")
            enriched.discard("Male")
            self._text_stated_attrs.discard("Male")
            self._text_stated_phrases.pop("Male", None)
            for attr in MALE_ONLY_ATTRS:
                forced_zero.add(attr)
                explicit_forced_zero.add(attr)
                enriched.discard(attr)
                self._text_stated_attrs.discard(attr)
                self._text_stated_phrases.pop(attr, None)

        # ---- Positive keyword scan ---------------------------------
        for keyword, attr in TEXT_POSITIVE_KEYWORDS.items():
            if _word_in(lower, keyword):
                # "old" is a negation for Young
                if keyword == "old":
                    forced_zero.add("Young")
                    explicit_forced_zero.add("Young")
                    enriched.discard("Young")
                    self._text_stated_attrs.discard("Young")
                    self._text_stated_phrases.pop("Young", None)
                else:
                    enriched.add(attr)
                    self._text_stated_attrs.add(attr)
                    # Store the phrase as it appeared in the input text
                    # (use keyword itself for multi-word ones, else attr phrase)
                    if attr not in self._text_stated_phrases:
                        self._text_stated_phrases[attr] = ATTR_TO_NATURAL_PHRASE.get(attr, keyword)

        # ---- Beard logic -------------------------------------------
        # "beard" → person HAS a beard → No_Beard = 0
        if _word_in(lower, "beard"):
            if _word_in(lower, "no beard"):
                enriched.add("No_Beard")
                self._text_stated_attrs.add("No_Beard")
                self._text_stated_phrases["No_Beard"] = "no beard"
            else:
                forced_zero.add("No_Beard")
                explicit_forced_zero.add("No_Beard")
                enriched.discard("No_Beard")
                self._text_stated_attrs.discard("No_Beard")
                self._text_stated_phrases.pop("No_Beard", None)

        # If specific facial hair is explicitly mentioned (e.g. "goatee"),
        # make sure the underlying "has beard" signal (No_Beard=0) is enabled
        # so `generate_description()` can actually surface the specific type.
        specific_facial_attrs = ["Goatee", "Mustache", "5_o_Clock_Shadow", "Sideburns"]
        if "No_Beard" in enriched:
            # "no beard" wins over specific facial hair mentions.
            for attr in specific_facial_attrs:
                if attr in enriched:
                    forced_zero.add(attr)
                    explicit_forced_zero.add(attr)
                    enriched.discard(attr)
                    self._text_stated_attrs.discard(attr)
                    self._text_stated_phrases.pop(attr, None)
        else:
            if any(attr in enriched for attr in specific_facial_attrs):
                forced_zero.add("No_Beard")
                # This is an implied constraint, not an explicit "beard" keyword.
                enriched.discard("No_Beard")
                self._text_stated_attrs.discard("No_Beard")
                self._text_stated_phrases.pop("No_Beard", None)

        # ---- Exclusive group cleanup (hair colour) -----------------
        for group in EXCLUSIVE_GROUPS:
            stated_in_group = [a for a in group if a in enriched]
            if len(stated_in_group) > 1:
                # Keep the last one mentioned (highest priority in text)
                keep = stated_in_group[-1]
                for a in stated_in_group:
                    if a != keep:
                        enriched.discard(a)
                        self._text_stated_attrs.discard(a)
                        self._text_stated_phrases.pop(a, None)

        # ---- Bald ↔ hair-style conflict ----------------------------
        if "Bald" in enriched:
            for a in HAIR_STYLE_ATTRS:
                enriched.discard(a)
                forced_zero.add(a)
                explicit_forced_zero.add(a)
                self._text_stated_attrs.discard(a)
                self._text_stated_phrases.pop(a, None)

        self._text_forced_zero_attrs = set(explicit_forced_zero)
        return list(enriched), list(forced_zero)

    # ------------------------------------------------------------------
    # Probability computation
    # ------------------------------------------------------------------

    def _build_effective_probs(
        self,
        predicted: Dict[str, float],
        text_matched: List[str],
        forced_zero: List[str],
        is_female: bool,
    ) -> Dict[str, float]:
        """
        Combine model predictions with hard constraints from text parsing.
        Applies gender-consistency rules before returning the final prob map.
        """
        text_matched_set = set(text_matched)
        forced_zero_set  = set(forced_zero)
        probs: Dict[str, float] = {}

        for attr in self.celeba_attrs:
            if attr in text_matched_set:
                probs[attr] = 1.0
            elif attr in forced_zero_set:
                probs[attr] = 0.0
            else:
                probs[attr] = float(predicted.get(attr, 0.0))

        # ---- Gender consistency ------------------------------------
        male_prob = probs.get("Male", 0.5)

        # Infer gender if not stated in text
        if is_female:
            male_prob = 0.0

        probs["Male"] = male_prob

        # If clearly female: zero out male-only attrs; boost female-only attrs
        if male_prob < 0.3:
            for attr in MALE_ONLY_ATTRS:
                if attr not in text_matched_set:
                    probs[attr] = 0.0
            # No_Beard is male-only; for females we just remove it
            probs["No_Beard"] = 0.0
            for attr in FEMALE_ONLY_ATTRS:
                if attr not in forced_zero_set and probs.get(attr, 0) < 0.3:
                    probs[attr] = max(probs.get(attr, 0.0), 0.4)

        # If clearly male: zero out female-only attrs that are nonsensical
        if male_prob >= 0.7:
            for attr in FEMALE_ONLY_ATTRS - {"Wearing_Earrings"}:
                if attr not in text_matched_set:
                    probs[attr] = 0.0
            # No_Beard default — if beard logic not resolved, give low-beard man
            # a reasonable No_Beard probability
            if "No_Beard" not in text_matched_set and "No_Beard" not in forced_zero_set:
                if probs.get("No_Beard", 0.0) < 0.1:
                    probs["No_Beard"] = 0.5   # ambiguous → sample it

        # ---- Bald ↔ hair-style conflict ----------------------------
        if probs.get("Bald", 0.0) > 0.7:
            for a in HAIR_STYLE_ATTRS:
                probs[a] = 0.0

        return probs

    # ------------------------------------------------------------------
    # Variation generation
    # ------------------------------------------------------------------

    def _select_variable_attrs(
        self,
        probs: Dict[str, float],
        text_matched_set: set,
    ) -> List[str]:
        """
        Return description-relevant attributes that are NOT text-stated
        and fall in the medium-confidence band (good candidates for
        combinatorial variation).
        """
        return [
            attr for attr in DESCRIPTION_ATTRS
            if attr in self.celeba_attrs
            and attr not in text_matched_set
            and MED_CONF <= probs.get(attr, 0.0) < self.high_thresh
        ]

    def _sample_single_vector(
        self,
        probs: Dict[str, float],
        text_matched_set: set,
        forced_zero_set: set,
        variable_overrides: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """
        Sample one attribute vector.

        variable_overrides lets callers fix specific uncertain attributes
        to predetermined values (used when building combinatorial variants).
        """
        overrides = variable_overrides or {}
        vector: Dict[str, int] = {}
        handled: set = set()

        # ---- Exclusive groups first --------------------------------
        for group in EXCLUSIVE_GROUPS:
            matched_in_group = [a for a in group if a in text_matched_set]
            override_in_group = {a: overrides[a] for a in group if a in overrides}

            if matched_in_group:
                for a in group:
                    vector[a] = 1 if a in matched_in_group else 0
            elif override_in_group:
                chosen = max(override_in_group, key=override_in_group.get)
                for a in group:
                    vector[a] = 1 if (a == chosen and override_in_group.get(a, 0) == 1) else 0
            else:
                group_probs = np.array([probs.get(a, 0.0) for a in group])
                none_prob   = max(0.0, 1.0 - group_probs.sum())
                all_probs   = np.append(group_probs, none_prob)
                all_probs   /= all_probs.sum()
                chosen_idx  = np.random.choice(len(all_probs), p=all_probs)
                for i, a in enumerate(group):
                    vector[a] = 1 if i == chosen_idx else 0

            handled.update(group)

        # ---- Remaining attributes ----------------------------------
        for attr in self.celeba_attrs:
            if attr in handled:
                continue
            if attr in forced_zero_set:
                vector[attr] = 0
                continue
            if attr in text_matched_set:
                vector[attr] = 1
                continue
            if attr in overrides:
                vector[attr] = overrides[attr]
                continue

            p = probs.get(attr, 0.0)
            if p >= self.high_thresh:
                vector[attr] = 1
            elif p <= self.low_thresh:
                vector[attr] = 0
            else:
                vector[attr] = int(np.random.binomial(1, p))

        return vector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_vectors(
        self,
        predicted_attrs: Dict[str, float],
        text_matched: List[str],
        input_text: str = "",
        num_variants: int = 4,
    ) -> List[Dict[str, int]]:
        """
        Generate diverse attribute vectors.

        Strategy
        --------
        1. Parse text → enrich text_matched, build forced_zero.
        2. Build effective probability map with gender-consistency rules.
        3. Identify "variable" attributes (medium-confidence, not text-stated).
        4. Create one *base* vector (deterministic high-conf attrs fixed).
        5. Generate additional variants by toggling combinations of
           variable attributes, ensuring each variant is unique.

        Parameters
        ----------
        predicted_attrs : BERT confidence scores {attr: float}
        text_matched    : attributes explicitly identified in input_text
        input_text      : raw user description string
        num_variants    : desired number of output vectors (best-effort)
        """
        # Step 1 — clean & parse
        predicted_attrs = {
            k: v for k, v in predicted_attrs.items() if k in self.celeba_attrs
        }
        is_female = any(
            _word_in(input_text.lower(), kw) for kw in FEMALE_KEYWORDS
        ) if input_text else False

        text_matched, forced_zero = self._parse_input_text(text_matched, input_text)
        text_matched_set = set(text_matched)
        forced_zero_set  = set(forced_zero)

        # Step 2 — effective probabilities
        probs = self._build_effective_probs(
            predicted_attrs, text_matched, forced_zero, is_female
        )

        # Step 3 — variable attributes (candidates for variation)
        variable_attrs = self._select_variable_attrs(probs, text_matched_set)
        # Sort by descending confidence so top candidates lead combinations
        variable_attrs.sort(key=lambda a: probs.get(a, 0.0), reverse=True)

        # Step 4 — base vector (no randomness for text-stated + high-conf attrs)
        base_vector = self._sample_single_vector(probs, text_matched_set, forced_zero_set)
        vectors = [base_vector]

        # If `variable_attrs` ends up empty (common when everything is either
        # text-stated or already in the high/low confidence bands), then
        # using only `variable_attrs` for deduplication would treat all sampled
        # candidates as identical, causing us to return only 1 variant.
        fingerprint_attrs = variable_attrs if variable_attrs else self.celeba_attrs

        seen: set = set()
        seen.add(_vec_fingerprint(base_vector, fingerprint_attrs))
        # Also track at the description level so variants that differ in
        # non-description attrs (e.g. obscure attrs) don't look identical.
        seen_desc: set = set()
        seen_desc.add(self.generate_description(base_vector))

        # Step 5 — combinatorial variants
        # Flip subsets of top-N variable attributes to create diverse vectors.
        top_vars = variable_attrs[:min(6, len(variable_attrs))]

        for r in range(1, len(top_vars) + 1):
            if len(vectors) >= num_variants:
                break
            for combo in combinations(range(len(top_vars)), r):
                if len(vectors) >= num_variants:
                    break
                overrides: Dict[str, int] = {}
                for idx in combo:
                    attr = top_vars[idx]
                    # Flip the base value of this attribute
                    overrides[attr] = 1 - base_vector.get(attr, 0)
                    # Resolve exclusive group conflicts caused by this flip
                    for group in EXCLUSIVE_GROUPS:
                        if attr in group and overrides[attr] == 1:
                            for other in group:
                                if other != attr:
                                    overrides[other] = 0

                candidate = self._sample_single_vector(
                    probs, text_matched_set, forced_zero_set, overrides
                )
                fp = _vec_fingerprint(candidate, fingerprint_attrs)
                desc = self.generate_description(candidate)
                if fp not in seen and desc not in seen_desc:
                    seen.add(fp)
                    seen_desc.add(desc)
                    vectors.append(candidate)

        # Fill remaining slots with stochastic samples if combos exhausted
        attempts = 0
        while len(vectors) < num_variants and attempts < 50:
            attempts += 1
            candidate = self._sample_single_vector(
                probs, text_matched_set, forced_zero_set
            )
            fp = _vec_fingerprint(candidate, fingerprint_attrs)
            desc = self.generate_description(candidate)
            if fp not in seen and desc not in seen_desc:
                seen.add(fp)
                seen_desc.add(desc)
                vectors.append(candidate)

        return vectors

    # ------------------------------------------------------------------
    # Tensor I/O
    # ------------------------------------------------------------------

    def vector_to_tensor(self, vector_dict: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(
            [vector_dict.get(attr, 0) for attr in self.celeba_attrs],
            dtype=torch.float32,
        )

    def save_tensors(self, vectors: List[Dict[str, int]], output_base: str) -> str:
        folder = os.path.join(
            output_base, datetime.now().strftime("input_%Y%m%d_%H%M%S")
        )
        os.makedirs(folder, exist_ok=True)
        for i, vec in enumerate(vectors):
            torch.save(
                self.vector_to_tensor(vec),
                os.path.join(folder, f"vector{i + 1}.pt"),
            )
        return folder

    # Backwards-compatible alias
    def save_vectors(self, vectors: List[Dict[str, int]], output_base: str) -> str:
        return self.save_tensors(vectors, output_base)

    # ------------------------------------------------------------------
    # Description generation
    # ------------------------------------------------------------------

    def generate_description(self, vector_dict: Dict[str, int]) -> str:
        """Convert an attribute vector to a natural-language description.

        Text-stated attributes (captured from the original input) always anchor
        the description.  Predicted / sampled attributes are appended as extras,
        so each variant shares the user's original terms but differs in the
        additional details inferred by the model.
        """
        original_stated       = getattr(self, "_text_stated_attrs",   set()) or set()
        original_forced_zero  = getattr(self, "_text_forced_zero_attrs", set()) or set()
        # Phrases as extracted from user input text, keyed by CelebA attr name.
        stated_phrases: Dict[str, str] = getattr(self, "_text_stated_phrases", {}) or {}

        is_male  = vector_dict.get("Male", 0) == 1
        # Determine gender word from input text first; fall back to vector.
        if "Male" in stated_phrases:
            gender = "man"
        elif any(kw in stated_phrases.values() for kw in ("woman", "girl", "lady")):
            gender = "woman"
        else:
            gender = "man" if is_male else "woman"

        # Check if the user said "woman/girl/lady" explicitly (Female keyword)
        _female_kw = FEMALE_KEYWORDS  # {"woman", "female", "girl", "lady", "she", "her"}
        is_female_stated = not is_male and ("Male" not in original_stated)

        # Age prefix -------------------------------------------------------
        if "Young" in stated_phrases:
            age_prefix = stated_phrases["Young"]          # "young" (literal from input)
        elif vector_dict.get("Young", 0) == 1:
            age_prefix = "young"
        elif "Young" in original_forced_zero and vector_dict.get("Young", 0) == 0:
            age_prefix = "older"
        else:
            age_prefix = ""

        # Bald prefix — "bald" is an adjective that belongs before the noun,
        # not as a "with bald" clause.
        is_bald = vector_dict.get("Bald", 0) == 1
        bald_prefix = "bald" if is_bald else ""

        # ------------------------------------------------------------------ 
        # Build two separate clause lists:
        #   base_clauses      → phrases from the user's original input text
        #   predicted_clauses → phrases from sampled/predicted attributes only
        # ------------------------------------------------------------------ 

        base_clauses:      List[str] = []
        predicted_clauses: List[str] = []

        def _add(attr: str, phrase: str) -> None:
            """Append phrase to the right clause list."""
            if attr in stated_phrases:
                base_clauses.append(stated_phrases[attr])
            elif attr in original_stated:
                base_clauses.append(phrase)
            else:
                predicted_clauses.append(phrase)

        # ---- Hair colour -------------------------------------------------
        hair_color_attrs = ["Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair"]
        hair_texture_attrs = ["Straight_Hair", "Wavy_Hair", "Bangs"]

        hair_color_phrase = {
            "Black_Hair":  "black hair",
            "Brown_Hair":  "brown hair",
            "Blond_Hair":  "blond hair",
            "Gray_Hair":   "gray hair",
        }
        hair_texture_phrase = {
            "Straight_Hair": "straight",
            "Wavy_Hair":     "wavy",
            "Bangs":         "with bangs",
        }

        active_color   = next((a for a in hair_color_attrs   if vector_dict.get(a, 0) == 1), None)
        # If the user stated a specific texture, always prefer that over any
        # predicted/sampled texture so the base phrase stays faithful to input.
        stated_texture = next((a for a in hair_texture_attrs if a in original_stated), None)
        active_texture = stated_texture or next(
            (a for a in hair_texture_attrs if vector_dict.get(a, 0) == 1), None
        )

        if vector_dict.get("Bald", 0) == 1:
            pass  # "bald" is handled as a subject adjective prefix, not a clause
        elif active_color or active_texture:
            # Compose a combined hair phrase:
            # "straight black hair" / "wavy hair" / "black hair with bangs" etc.
            color_str   = hair_color_phrase.get(active_color, "")   if active_color   else ""
            texture_str = hair_texture_phrase.get(active_texture, "") if active_texture else ""

            if texture_str == "with bangs":
                # e.g. "black hair with bangs"
                combined = (f"{color_str} with bangs" if color_str else "hair with bangs")
            elif texture_str:
                # e.g. "wavy black hair" / "straight hair"
                # color_str already ends with " hair", so just prepend texture
                color_base = color_str.replace(" hair", "").strip() if color_str else ""
                combined = f"{texture_str} {color_base} hair".strip() if color_base else f"{texture_str} hair"
            else:
                combined = color_str  # plain "black hair"

            # Determine which bucket this hair phrase belongs to.
            # If ANY of the involved attrs was stated in input → base clause.
            involved = [a for a in ([active_color] if active_color else []) +
                                    ([active_texture] if active_texture else [])
                        if a is not None]
            if any(a in original_stated for a in involved):
                base_clauses.append(combined)
            else:
                predicted_clauses.append(combined)

        # ---- Facial hair (male only) -------------------------------------
        if is_male or "Male" in original_stated:
            no_beard_val  = vector_dict.get("No_Beard", 1)
            beard_present = (no_beard_val == 0)

            if beard_present:
                if "No_Beard" in original_forced_zero:
                    # user explicitly mentioned "beard" → anchor it
                    base_clauses.append("a beard")
                specific_beard_map = {
                    "Goatee":           "a goatee",
                    "Mustache":         "a mustache",
                    "5_o_Clock_Shadow": "light stubble",
                    "Sideburns":        "sideburns",
                }
                for attr, phrase in specific_beard_map.items():
                    if vector_dict.get(attr, 0) == 1:
                        _add(attr, phrase)
            else:
                # no beard — only mention if user stated it
                if "No_Beard" in original_stated:
                    base_clauses.append("no beard")

        # ---- Face shape + facial features --------------------------------
        face_feature_order = [
            "High_Cheekbones", "Big_Nose", "Narrow_Eyes",
            "Bushy_Eyebrows", "Arched_Eyebrows",
            "Pale_Skin", "Rosy_Cheeks",
            "Chubby", "Double_Chin", "Smiling",
            "Oval_Face",
        ]
        face_feature_phrase = {
            "High_Cheekbones":  "high cheekbones",
            "Big_Nose":         "a prominent nose",
            "Narrow_Eyes":      "narrow eyes",
            "Bushy_Eyebrows":   "bushy eyebrows",
            "Arched_Eyebrows":  "arched eyebrows",
            "Pale_Skin":        "pale skin",
            "Rosy_Cheeks":      "rosy cheeks",
            "Chubby":           "a chubby face",
            "Double_Chin":      "a double chin",
            "Smiling":          "a smile",
            "Oval_Face":        "an oval face",
        }

        for attr in face_feature_order:
            if attr == "Oval_Face":
                if not (self._include_face_shape and vector_dict.get("Oval_Face", 0) == 1):
                    continue
            else:
                if vector_dict.get(attr, 0) != 1:
                    continue
            phrase = face_feature_phrase.get(attr)
            if phrase:
                _add(attr, phrase)

        # ---- Accessories -------------------------------------------------
        accessory_order = [
            "Eyeglasses", "Wearing_Hat", "Wearing_Necktie",
            "Wearing_Necklace", "Wearing_Earrings",
            "Wearing_Lipstick", "Heavy_Makeup",
        ]
        accessory_phrase = {
            "Eyeglasses":       "glasses",
            "Wearing_Hat":      "a hat",
            "Wearing_Necktie":  "a necktie",
            "Wearing_Necklace": "a necklace",
            "Wearing_Earrings": "earrings",
            "Wearing_Lipstick": "makeup",
            "Heavy_Makeup":     "makeup",
        }

        base_acc_set:      set = set()
        predicted_acc_set: set = set()
        for attr in accessory_order:
            if vector_dict.get(attr, 0) != 1:
                continue
            phrase = accessory_phrase[attr]
            if attr in original_stated:
                base_acc_set.add(phrase)
            else:
                predicted_acc_set.add(phrase)

        acc_order_stable = ["glasses", "a hat", "a necktie", "a necklace", "earrings", "makeup"]
        base_accessories      = [p for p in acc_order_stable if p in base_acc_set]
        predicted_accessories = [p for p in acc_order_stable
                                  if p in predicted_acc_set and p not in base_acc_set]

        if base_accessories:
            base_clauses.append(and_join(base_accessories))
        if predicted_accessories:
            predicted_clauses.append(and_join(predicted_accessories))

        # ---- Assemble final description ----------------------------------
        base_parts = [p for p in [age_prefix, bald_prefix, gender] if p]
        subject    = " ".join(base_parts)

        all_clauses = base_clauses + predicted_clauses
        if all_clauses:
            description = subject + " with " + ", ".join(all_clauses)
        else:
            description = subject

        return description.capitalize()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vec_fingerprint(vector: Dict[str, int], attrs: List[str]) -> Tuple[int, ...]:
    """Compact fingerprint of a vector for deduplication (variable attrs only)."""
    return tuple(vector.get(a, 0) for a in attrs)


def and_join(items: List[str]) -> str:
    """Join a list of strings with commas and 'and' before the last item."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]