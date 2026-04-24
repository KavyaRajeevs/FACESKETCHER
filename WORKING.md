
## Module 1 – Multi‑Attribute Face Sampler

This module trains a BERT‑based multi‑label classifier on CelebA attributes, exposes it via a FastAPI service, and uses a rule‑aware sampler plus a Streamlit UI to generate diverse, gender‑consistent facial‑attribute vectors and human‑readable descriptions that can drive downstream face generation.

---

## High‑Level Pipeline

- **Step 1 – Train BERT on CelebA attributes (`module1/train.py`)**  
  - Convert each CelebA attribute row into a short text like `"young blond hair smiling"` and train a **multi‑label classifier** using Hugging Face **`bert-base-uncased`** (`BertTokenizer`, `BertForSequenceClassification`) implemented in PyTorch.  
  - The model learns to map an attribute description back to a probability for every CelebA attribute (multi‑label, sigmoid outputs).  
  - Outputs: a fine‑tuned BERT model (weights + tokenizer) saved in `./models/bert_celeba` and an `attributes.csv` file listing attribute names.

- **Step 2 – Attribute prediction API with BERT (`module1/app.py`)**  
  - Wrap the trained **`bert-base-uncased`** model in a **FastAPI** service exposing a `/predict` endpoint.  
  - Given free‑text like `"a young woman with blond hair and glasses"`, the API tokenizes the text with `BertTokenizer`, runs `BertForSequenceClassification` on GPU/CPU, and returns:  
    - `predicted_attributes`: per‑attribute confidence scores (after sigmoid, optionally temperature‑scaled).  
    - `text_matched_attributes`: attributes explicitly mentioned in the input text (simple string matching boosts those scores slightly).  
    - `top_attributes`: the top‑N attributes by confidence for quick inspection.  
  - The prediction JSON is saved both locally (`./outputs`) and to an external folder so it can be uploaded into the Streamlit UI or used by other tools.

- **Step 3 – Multi‑Attribute Sampler (`module1/sampler.py`)**  
  - Take BERT’s confidence scores plus the attributes explicitly mentioned in the text and turn them into **multiple, diverse but realistic attribute vectors**.  
  - Apply domain rules (mutually exclusive hair colors, gender‑specific attributes, bald vs hair style, etc.) to ensure generated combinations are coherent and match the textual description.  
  - Convert vectors to **PyTorch tensors** (`torch.tensor([...], dtype=float32)`) saved on disk, and generate **natural‑language descriptions** of each variant like `"Young woman with wavy brown hair, glasses, and high cheekbones"`.

---

## Models Used and Why

- **BERT tokenizer & encoder (`bert-base-uncased`) – text → attributes**  
  - Files: `train.py`, `app.py`.  
  - Components: `BertTokenizer`, `BertForSequenceClassification` with `problem_type="multi_label_classification"`.  
  - **Why BERT?**  
    - Pretrained on large English corpora; understands short attribute phrases and natural language prompts.  
    - Supports **multi‑label classification**: we need independent probabilities for ~40 facial attributes, not a single class.  
    - Off‑the‑shelf implementation in `transformers` keeps the code compact and easy to deploy.

- **Multi‑Attribute Sampler – probabilistic + rule‑based post‑processing**  
  - File: `sampler.py`.  
  - **Input model output**: dictionary `{attribute_name: probability}` from BERT, plus attributes that were explicitly matched in text.  
  - **Why a custom sampler on top of BERT?**  
    - Raw BERT probabilities may violate domain constraints (e.g., both `Blond_Hair` and `Black_Hair` near 0.5; beards on clearly female descriptions).  
    - We want **multiple plausible variants** around the same description (for creative exploration) instead of one hard thresholded vector.  
    - By combining probabilities with hard rules and controlled randomness, we can generate diverse, realistic, and text‑consistent attribute sets.

---

## File‑by‑File Details

### `module1/train.py` – Training the BERT Attribute Classifier

- **Core logic**  
  - Load CelebA attributes CSV (`./data/list_attr_celeba.csv`), drop `image_id` if present, and convert labels from \(-1, 1\) to \((0, 1)\).  
  - Subsample to 5k rows for faster CPU training.  
  - Convert each attribute row into a space‑separated description using `attributes_to_text`, keeping only attributes with value `1` (true) and replacing underscores with spaces.  
  - Split texts/labels into train and validation sets (scikit‑learn `train_test_split`).  
  - Tokenize with **`BertTokenizer.from_pretrained("bert-base-uncased")`** using padding/truncation to `MAX_LEN`.  
  - Wrap data in a custom `CelebADataset` and PyTorch `DataLoader`.  
  - Instantiate **`BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=..., problem_type="multi_label_classification")`** and optimize with `AdamW`.  
  - Train for a small number of epochs, accumulating average loss, then save:  
    - Model weights + tokenizer to `./models/bert_celeba`.  
    - `attributes.csv` listing attribute names (used later by the API and sampler).

- **Why this design**  
  - Treating attribute strings as text means BERT can reuse its language understanding rather than learning from scratch.  
  - Multi‑label setup lets us model correlations (e.g., hair color + beard) while still predicting each attribute independently.  
  - Keeping training simple (single file, small sample, single epoch) makes iteration fast while still producing reasonable attribute probabilities.

### `module1/app.py` – FastAPI Prediction and Sampler Orchestration

- **Core logic**  
  - On startup (`lifespan` context):  
    - Load the fine‑tuned **BERT model and tokenizer** from `MODEL_PATH = "./models/bert_celeba"`.  
    - Load attribute names from `attributes.csv`.  
    - Initialize a single `MultiAttributeSampler` instance with the attributes CSV.  
  - `/predict` endpoint:  
    - Validate non‑empty text.  
    - Tokenize input with BERT’s tokenizer and run the classifier on `DEVICE` (CPU/GPU).  
    - Apply sigmoid to logits (with a configurable temperature for softer probabilities) to get attribute confidences.  
    - For each attribute:  
      - If the attribute name (converted to lower‑cased words) appears in the input text, slightly **boost the score** and add the attribute to `text_matched_attributes`.  
    - Sort attributes by confidence, pick `top_attributes` for convenience, and serialize everything to a timestamped JSON file in both `./outputs` and `EXTERNAL_OUTPUT_PATH`.  
  - **Sampler integration**  
    - Pass `predicted_attributes`, `text_matched_attributes`, and the original text to `MultiAttributeSampler.sample_vectors` with `NUM_VARIANTS` (default 4).  
    - Save the resulting attribute tensors under `SAMPLER_OUTPUT_PATH`.  
    - For each sampled vector, call `MultiAttributeSampler.generate_description` to produce a readable caption.  
    - Return both the **raw predictions** and **sampler variants** (attributes + description) in the `PredictResponse`.

- **Why this design**  
  - FastAPI keeps the service lightweight and easy to integrate with other applications (e.g., a front‑end or another module generating faces).  
  - Centralizing model loading in the lifespan handler avoids reloading BERT per request, which would be too slow.  
  - Returning both the continuous scores and discrete sampled variants supports analytical use (inspecting probabilities) and creative use (multiple concrete attribute sets).

### `module1/sampler.py` – Multi‑Attribute Sampler and Description Generator

- **Domain rules**  
  - **Mutually exclusive groups** (`EXCLUSIVE_GROUPS`): enforce that exactly one of `["Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair", "Bald"]` (hair color group) and one of `["Straight_Hair", "Wavy_Hair", "Bangs"]` is active at most, including an implicit “none” state.  
  - **Gender‑specific attributes**:  
    - `MALE_ONLY_ATTRS` (beards, sideburns, necktie) and `FEMALE_ONLY_ATTRS` (earrings, lipstick, heavy makeup, etc.).  
    - Female‑coded text zeroes out male‑only attributes; male‑coded cases suppress some female‑only attributes.  
  - **Bald vs hair style**: if `Bald` is active, all hair‑style attributes are forced to 0.  
  - **Description focus** (`DESCRIPTION_ATTRS`): only a subset of attributes is used to build natural‑language descriptions (e.g., hair, beard, glasses, hat, facial features).

- **Main class: `MultiAttributeSampler`**  
  - **Initialization**  
    - Load attribute names from a CSV (same list as used by BERT).  
  - **Text parsing (`_parse_input_text`)**  
    - Detect gender keywords (`"woman"`, `"girl"`, `"man"`, `"boy"`, etc.).  
    - Apply keyword‑to‑attribute mappings (e.g., `"blond"` → `Blond_Hair`, `"glasses"` → `Eyeglasses`).  
    - Resolve conflicts in exclusive groups based on text (“last mention wins”), enforce beard logic (`"no beard"` vs `"beard"`), and set attributes that must be zero (`forced_zero`).  
  - **Probability adjustment (`_build_effective_probs`)**  
    - Merge BERT probabilities with hard constraints from text parsing.  
    - Infer gender probability and use it to:  
      - Zero out nonsensical gendered attributes.  
      - Optionally boost plausible gender‑consistent attributes to a minimum probability.  
    - Resolve `Bald` vs hair‑style conflicts at the probability level before sampling.  
  - **Vector sampling (`_sample_single_vector`, `sample_vectors`)**  
    - For each exclusive group, either:  
      - Obey explicit text / overrides, or  
      - Sample one member or “none” according to probabilities.  
    - For other attributes:  
      - Always‑on if above `high_thresh`.  
      - Always‑off if below `low_thresh`.  
      - Otherwise sample from a Bernoulli with probability `p`.  
    - To create **multiple variants**, identify “medium‑confidence, description‑relevant” attributes and systematically flip subsets of them, deduplicating with a fingerprint to ensure each variant is unique.  
  - **Tensor saving and descriptions**  
    - `vector_to_tensor` / `save_tensors`: turn attribute dicts into PyTorch tensors and save them with timestamped folder names.  
    - `generate_description`:  
      - Build phrases for age, gender, hair color/texture, facial hair, face shape, notable facial features, and accessories.  
      - Return a natural sentence like `"Young man with wavy brown hair, glasses, and high cheekbones"`.

---

## End‑to‑End Flow Summary

1. **Train** the BERT‑based multi‑label classifier on CelebA attributes using `train.py`.  
2. **Serve** the trained BERT model via FastAPI in `app.py`, exposing `/predict` that transforms free‑text into per‑attribute confidence scores and saves JSON outputs.  
3. **Sample** multiple realistic attribute vectors from those scores with `MultiAttributeSampler` (`sampler.py`), enforcing domain rules and generating natural‑language descriptions.  
4. **Explore** and debug the sampler outputs interactively in the Streamlit UI (`multiui.py`) by uploading JSON files produced by the API or by other tools.



