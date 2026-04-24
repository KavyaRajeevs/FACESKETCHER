# LAGAN (Text → Sketch) API

This folder contains a small Flask API wrapper around a **LAGAN-style text-conditioned GAN** that generates a **face sketch image** from a natural-language description.

## What each file/folder does

### `app.py`
- **Purpose**: Runs a Flask server and exposes HTTP endpoints to generate sketches.
- **Key parts**:
  - **CORS config**: allows requests from `http://localhost:3000`.
  - **Model paths**: hard-coded absolute Windows paths for:
    - text encoder weights (`text_encoder600.pth`)
    - generator weights (`netG_epoch_600.pth`)
    - YAML config (`train_sketch_18_4.yml`)
  - **Routes**:
    - `GET /`: simple “API is running” message.
    - `POST /generate_sketch`: expects JSON `{ "description": "..." }`, generates an image, and returns a PNG via `send_file`.
    - `GET /health`: health check + timestamp.
  - **Important note**: the output directory is also hard-coded to a `C:\Users\HP\Desktop\FaceFinder\...` path. If that path doesn’t exist on your machine, generation will fail until you update it.

### `text_to_sketch.py`
- **Purpose**: Inference wrapper used by `app.py`.
- **Core class**: `TextToSketchGenerator`
  - **Loads config** via `cfg_from_file()` (from `config.py`) if a YAML path is provided.
  - **Loads vocabulary** from `captions_org.pickle` (currently hard-coded to an absolute path).
  - **Loads models** (CPU-only):
    - `RNN_ENCODER` (text encoder) from `model.py`
    - `G_NET` (generator) from `model.py`
  - **Text preprocessing**:
    - tokenizes with `RegexpTokenizer(r'\w+')`
    - maps tokens to IDs via `wordtoix` loaded from the pickle
    - errors if none of the words are in the vocabulary
  - **Generation**:
    - builds word embeddings + sentence embedding using the text encoder
    - samples Gaussian noise \(z\)
    - calls the generator: `fake_imgs, _, _, _ = netG(z, sent_emb, words_embs, mask)`
    - uses the final branch output (`fake_imgs[-1]`) and converts it to a PIL image in uint8

### `model.py`
- **Purpose**: Defines the neural network modules used by LAGAN.
- **Main components**:
  - **`RNN_ENCODER`**: embedding + RNN (LSTM/GRU) that produces:
    - `words_emb`: per-token features (used for attention)
    - `sent_emb`: a global sentence embedding
  - **`CA_NET`**: conditioning augmentation network that turns `sent_emb` into:
    - `c_code` (sampled conditioning vector) and `(mu, logvar)` (for KL regularization during training)
  - **`G_NET`**: multi-branch generator (controlled by `cfg.TREE.BRANCH_NUM`), typically producing progressively larger images (e.g., 64→128→256). Each “next stage” uses attention over word embeddings.
  - **Attention usage**: `NEXT_STAGE_G` uses `SpatialAttention` and `ChannelAttention` from `attention.py`.
  - **Also includes** discriminator definitions (`D_NET64/128/256`) which are used for training, not needed for inference in this API.
- **Device behavior**: many modules explicitly call `.to('cpu')` inside constructors and forward passes, so this code is currently CPU-oriented.

### `attention.py`
- **Purpose**: Implements attention blocks used inside the generator refinement stages.
- **Key functions/classes**:
  - **`func_attention(query, context, gamma1)`**: computes soft attention between query features and context features.
  - **`SpatialAttention`**: attends over **words** for each **spatial location** in the feature map; supports a token mask.
  - **`ChannelAttention`**: computes an additional channel-wise attention pathway conditioned on word features.
- **Device behavior**: this file also forces tensors to CPU in forward passes (`.to('cpu')`).

### `ponnu_models/`
- **Purpose**: Stores trained model checkpoint files used for inference.
- **Current contents**:
  - `text_encoder600.pth`: weights for `RNN_ENCODER`
  - `netG_epoch_600.pth`: weights for `G_NET`
  - `image_encoder600.pth`: likely a `CNN_ENCODER` checkpoint (not used by `text_to_sketch.py` during inference)

## How the “LAGAN module” works (end-to-end)

1. **Client sends text** to the API:
   - `POST /generate_sketch` with JSON `{ "description": "..." }`
2. **Tokenization & vocabulary lookup**:
   - text → tokens → token IDs (via `captions_org.pickle`)
3. **Text encoding** (`RNN_ENCODER`):
   - produces `words_embs` (token-level) and `sent_emb` (sentence-level)
4. **Conditioning augmentation** (`CA_NET`):
   - turns `sent_emb` into `c_code` (a stochastic conditioning vector)
5. **Image generation** (`G_NET`):
   - samples noise `z`
   - generates images across branches (progressive refinement)
   - uses attention over `words_embs` inside later stages
6. **Post-processing**:
   - takes final image tensor, rescales from `[-1, 1]` to `[0, 255]`, converts to PNG
7. **API returns the PNG** as the HTTP response.

## Running the API locally

### 1) Create a virtual environment and install dependencies

If you already have a Python environment, install at least:

- `flask`
- `flask-cors`
- `torch`
- `torchvision`
- `pillow`
- `nltk`
- `numpy`
- `pyyaml`
- `easydict`

Example:

```bash
pip install flask flask-cors torch torchvision pillow nltk numpy pyyaml easydict
```

### 2) Fix hard-coded paths (required on most machines)

Several paths in `app.py` and `text_to_sketch.py` are absolute and point to `C:\Users\HP\Desktop\FaceFinder\...`.

Update them to your local paths, preferably relative to this folder:

- **Weights**: `api/ponnu_models/text_encoder600.pth`, `api/ponnu_models/netG_epoch_600.pth`
- **Config**: `api/config/train_sketch_18_4.yml`
- **Vocabulary**: `api/captions_org.pickle`
- **Output**: `api/generated_sketches/`

### 3) Start the server

From the `api/` directory:

```bash
python app.py
```

The API will run on `http://127.0.0.1:5001`.

## API usage

### Generate sketch

Request:

- **Method**: `POST`
- **Path**: `/generate_sketch`
- **Body**:

```json
{
  "description": "young man with short hair and glasses"
}
```

Response:

- **200**: PNG image bytes
- **400**: missing `description`
- **500**: model/vocab/path/runtime error

## Notes / common issues

- **CPU-only**: the code forces CPU usage in multiple places; generation may be slow.
- **Vocabulary mismatch**: `text_to_sketch.py` includes logic to resize the text encoder if checkpoint vocab size differs from `captions_org.pickle`.
- **NLTK tokenizer**: uses `RegexpTokenizer`, so you don’t need to download NLTK corpora for tokenization.

