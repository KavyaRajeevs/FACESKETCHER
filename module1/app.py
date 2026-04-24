from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import os
import json
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from sampler import MultiAttributeSampler

# =========================
# Paths
# =========================
MODEL_PATH = "./models/bert_celeba"
OUTPUT_PATH = "./outputs"
EXTERNAL_OUTPUT_PATH = "C:/Users/hp/OneDrive/Desktop/Multi-attribute-sampler/input"
SAMPLER_OUTPUT_PATH = r"C:\Users\hp\OneDrive\Desktop\FaceSketcher\Module1\module1\multi-outputs"
ATTRIBUTES_CSV_PATH = f"{MODEL_PATH}/attributes.csv"
NUM_VARIANTS = 4

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(EXTERNAL_OUTPUT_PATH, exist_ok=True)
os.makedirs(SAMPLER_OUTPUT_PATH, exist_ok=True)

# =========================
# Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Temperature for softer probabilities
TEMPERATURE = 1.5

# Simple gender keyword sets used to post-process BERT scores
FEMALE_KEYWORDS = {"woman", "female", "girl", "lady", "she", "her"}
MALE_KEYWORDS   = {"man", "male", "boy", "he", "him", "his"}

# Attributes that are inappropriate / illogical for males
FEMALE_ONLY_ATTRIBUTES = {
    "Heavy_Makeup",
    "Lipstick",
    "Wearing_Lipstick",       # alternate name some CelebA CSVs use
    "Heavy_Makeup",
}

# Keyword → attributes that should be suppressed when the keyword appears in text
# e.g. "beard" strongly implies the person has a beard → No_Beard should be suppressed
KEYWORD_SUPPRESSION_MAP: Dict[str, List[str]] = {
    "beard":     ["No_Beard"],
    "bearded":   ["No_Beard"],
    "no beard":  ["Goatee", "Mustache", "Sideburns"],   # opposite case
    "mustache":  ["No_Beard"],
    "goatee":    ["No_Beard"],
}

# Threshold below which an attribute is considered "not predicted"
SUPPRESSION_SCORE = 0.10   # drive suppressed attributes down to this value

# Global model / sampler variables
tokenizer = None
model     = None
attributes = None
sampler: Optional[MultiAttributeSampler] = None


class SamplerVariant(BaseModel):
    attributes: Dict[str, int]
    description: str

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    input_text: str
    predicted_attributes: dict
    top_attributes: list
    text_matched_attributes: list
    sampler_variants: Optional[List[SamplerVariant]] = None
    sampler_output_folder: Optional[str] = None


def apply_attribute_constraints(
    results: Dict[str, float],
    text_lower: str,
    is_male: bool,
    is_female: bool,
) -> Dict[str, float]:
    """
    Post-process predicted attribute scores to enforce logical and
    gender-based constraints.

    Rules applied (in order):
    1. Male subjects → suppress female-only attributes (makeup, lipstick …)
    2. Beard / facial-hair keywords → suppress No_Beard and vice-versa
    3. Any other keyword-driven suppressions defined in KEYWORD_SUPPRESSION_MAP
    """

    # ── Rule 1: gender constraints ────────────────────────────────────────────
    if is_male and not is_female:
        for attr in FEMALE_ONLY_ATTRIBUTES:
            if attr in results:
                results[attr] = SUPPRESSION_SCORE

    # ── Rule 2 & 3: keyword-driven suppressions ───────────────────────────────
    for keyword, attrs_to_suppress in KEYWORD_SUPPRESSION_MAP.items():
        if keyword in text_lower:
            for attr in attrs_to_suppress:
                if attr in results:
                    results[attr] = SUPPRESSION_SCORE

    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, attributes, sampler
    # Load model
    tokenizer  = BertTokenizer.from_pretrained(MODEL_PATH)
    model      = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    attributes = pd.read_csv(ATTRIBUTES_CSV_PATH, header=None)[0].tolist()
    model.to(DEVICE)
    model.eval()

    # Initialize multi-attribute sampler
    sampler = MultiAttributeSampler(ATTRIBUTES_CSV_PATH)

    yield
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )

    # Move once to DEVICE; non_blocking helps a bit on CUDA with pinned memory
    inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits / TEMPERATURE
        probs   = torch.sigmoid(logits).cpu().numpy()[0]

    # ── Detect gender from raw text ───────────────────────────────────────────
    text_lower = text.lower()
    is_female  = any(kw in text_lower for kw in FEMALE_KEYWORDS)
    is_male    = any(kw in text_lower for kw in MALE_KEYWORDS)

    # ── Build initial results + track text-matched attributes ─────────────────
    results      : Dict[str, float] = {}
    text_matched : List[str]        = []

    for attr, score in zip(attributes, probs):
        clean_attr = attr.replace("_", " ").lower()

        # If attribute mentioned in text → boost slightly
        if clean_attr in text_lower:
            score = min(score + 0.15, 1.0)
            text_matched.append(attr)

        results[attr] = round(float(score), 3)

    # ── Apply logical / gender constraints ────────────────────────────────────
    results = apply_attribute_constraints(results, text_lower, is_male, is_female)

    # Round again after constraint adjustments
    results = {k: round(v, 3) for k, v in results.items()}

    # Sort by confidence (descending)
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # Top attributes (exclude the raw index column "0" if present)
    top_attrs = [item for item in list(results.items())[:10] if item[0] != "0"]

    # ── Save output JSON ──────────────────────────────────────────────────────
    timestamp            = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file          = os.path.join(OUTPUT_PATH,          f"attributes_{timestamp}.json")
    external_output_file = os.path.join(EXTERNAL_OUTPUT_PATH, f"attributes_{timestamp}.json")

    output_data = {
        "input_text":             text,
        "text_matched_attributes": text_matched,
        "predicted_attributes":   results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    with open(external_output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    # ── Run Multi-Attribute Sampler ───────────────────────────────────────────
    sampler_variants    : List[SamplerVariant] = []
    sampler_output_folder: Optional[str]       = None

    if sampler is not None:
        try:
            vectors = sampler.sample_vectors(
                predicted_attrs=results,
                text_matched=text_matched,
                input_text=text,
                num_variants=NUM_VARIANTS,
            )
            sampler_output_folder = sampler.save_tensors(vectors, SAMPLER_OUTPUT_PATH)
            for vec in vectors:
                description = sampler.generate_description(vec)
                sampler_variants.append(
                    SamplerVariant(attributes=vec, description=description)
                )
        except Exception as e:
            # If sampler fails, log and continue returning base prediction
            print(f"MultiAttributeSampler error: {e}")

    return PredictResponse(
        input_text=text,
        predicted_attributes=results,
        top_attributes=top_attrs,
        text_matched_attributes=text_matched,
        sampler_variants=sampler_variants or None,
        sampler_output_folder=sampler_output_folder,
    )