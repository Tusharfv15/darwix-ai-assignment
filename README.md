# 🎙️ Empathy Engine

> **Text → Emotion → Expressive Voice** — An AI pipeline that detects emotion in text and synthesizes emotionally-aware speech using Parler-TTS Large, deployed on Modal serverless GPUs.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Design Choices](#design-choices)
  - [Emotion Detection](#emotion-detection)
  - [Emotion-to-Voice Mapping](#emotion-to-voice-mapping)
  - [Intensity Scaling](#intensity-scaling)
  - [Why Parler-TTS over SSML](#why-parler-tts-over-ssml)
  - [Why Modal](#why-modal)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Modal Deployment](#modal-deployment)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Bonus Features](#bonus-features)

---

## Project Overview

Standard Text-to-Speech systems are functional but robotic — they lack prosody, emotional range, and the subtle vocal cues that build trust and rapport. The **Empathy Engine** bridges this gap by:

1. Detecting the **emotion and intensity** of input text using a fine-tuned transformer model
2. **Mapping** that emotion to a natural language voice description
3. Synthesizing **expressive, human-like speech** using Parler-TTS Large

The result is audio that genuinely sounds different depending on whether the text is joyful, frustrated, fearful, or neutral — not just faster or slower, but expressively different.

---

## Architecture

```
User Input (text)
        │
        ▼
┌───────────────────────────────────────────────┐
│              EmpathyEngine (Modal A10G)       │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │  Step 1 — Emotion Detection             │  │
│  │  j-hartmann/emotion-english-            │  │
│  │  distilroberta-base                     │  │
│  │  → { emotion, intensity, all_scores }   │  │
│  └──────────────────┬──────────────────────┘  │
│                     │                         │
│  ┌──────────────────▼──────────────────────┐  │
│  │  Step 2 — Voice Mapping (local logic)   │  │
│  │  emotion + intensity → tier (low/mid/   │  │
│  │  high) → Parler-TTS description string  │  │
│  └──────────────────┬──────────────────────┘  │
│                     │                         │
│  ┌──────────────────▼──────────────────────┐  │
│  │  Step 3 — TTS Synthesis                 │  │
│  │  parler-tts/parler-tts-large-v1         │  │
│  │  → output_<timestamp>.wav               │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
        │
        ▼
  FileResponse (.wav)
  + Emotion metadata in HTTP headers
        │
        ▼
  Streamlit Frontend
  (audio player + confidence score bars)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Emotion Detection | `j-hartmann/emotion-english-distilroberta-base` via HuggingFace Transformers |
| TTS Model | `parler-tts/parler-tts-large-v1` |
| GPU Compute | Modal serverless A10G |
| Model Caching | Modal Volumes |
| API | Modal `@modal.fastapi_endpoint` |
| Frontend | Streamlit |

---

## Design Choices

### Emotion Detection

The model used is [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) — a DistilRoBERTa model fine-tuned on emotion classification. It returns scores across **7 emotion labels**:

| Label | Description |
|---|---|
| `joy` | Happiness, enthusiasm, excitement |
| `anger` | Frustration, rage, hostility |
| `sadness` | Grief, melancholy, sorrow |
| `fear` | Anxiety, nervousness, dread |
| `surprise` | Shock, astonishment, unexpectedness |
| `disgust` | Contempt, revulsion, disapproval |
| `neutral` | Flat, informational, no strong emotion |

The model returns a confidence score for each label. The **dominant emotion** (highest score) is used for voice mapping, and its score becomes the **intensity** value used for tier classification.

```python
raw        = self.classifier(text)[0]   # list of {label, score}
raw_sorted = sorted(raw, key=lambda x: x["score"], reverse=True)
dominant   = raw_sorted[0]

result = {
    "emotion":    dominant["label"],       # e.g. "joy"
    "intensity":  round(dominant["score"], 4),  # e.g. 0.94
    "all_scores": {item["label"]: round(item["score"], 4) for item in raw_sorted},
}
```

---

### Emotion-to-Voice Mapping

Rather than manipulating raw audio parameters (pitch, rate, volume) after the fact, the Empathy Engine uses **Parler-TTS's native description-based control**. The voice mapper translates the detected emotion into a natural language prompt that Parler-TTS uses to condition its generation.

Each emotion has **3 tiers** (low / mid / high) based on intensity:

```python
# Intensity thresholds
LOW  = 0.5   # intensity < 0.5  → low tier
HIGH = 0.8   # intensity >= 0.8 → high tier
             # 0.5 <= intensity < 0.8 → mid tier
```

Full mapping table:

| Emotion | Low | Mid | High |
|---|---|---|---|
| `joy` | Cheerful, friendly, slightly upbeat | Enthusiastic, warm, faster pace | Very enthusiastic, high energy, fast, expressive |
| `anger` | Firm, direct, slightly clipped | Sharp, tense, fast, forceful | Very intense, aggressive, rapid, harsh |
| `sadness` | Slightly subdued, slow | Soft, slow, quiet, melancholic | Very slow, quiet, heavy, sorrowful |
| `fear` | Slightly nervous, hesitant | Anxious, unsteady, tense | Very fearful, trembling, erratic, whispered |
| `surprise` | Mildly surprised, alert | Excited, fast, bright, animated | Very shocked, rapid, breathless, expressive |
| `disgust` | Flat, cold, detached | Cold, dismissive, dry, contemptuous | Very cold, slow, contemptuous, deliberate |
| `neutral` | Calm, clear, moderate pace (same across all tiers) | | |

Every description also appends the Parler-TTS recommended quality suffix:

```python
QUALITY_SUFFIX = "The recording is of very high quality, with the speaker's voice sounding clear and very close up."
description = f"{VOICE_MAP[emotion][tier]} {QUALITY_SUFFIX}"
```

This suffix is recommended by the Parler-TTS authors to consistently produce clean, high-fidelity output.

---

### Intensity Scaling

The same emotion detected at different confidence levels produces meaningfully different audio. For example:

```
Input: "Good."
→ emotion: joy, intensity: 0.42 (low)
→ "A cheerful and friendly voice with a slightly upbeat tone at a moderate pace."

Input: "THIS IS THE BEST DAY OF MY LIFE!!!"
→ emotion: joy, intensity: 0.97 (high)
→ "A very enthusiastic, high energy voice with a fast pace, warm tone, and expressive delivery."
```

This directly satisfies the **Intensity Scaling** bonus requirement from the problem statement.

---

### Why Parler-TTS over SSML

The problem statement suggests SSML (Speech Synthesis Markup Language) as a bonus approach. Instead, Parler-TTS was chosen because:

| | SSML (e.g. Google Cloud TTS) | Parler-TTS (our approach) |
|---|---|---|
| Control mechanism | XML tags (`<prosody rate="fast">`) | Natural language descriptions |
| Expressiveness | Limited to rate, pitch, volume | Full voice character, timbre, energy |
| Cost | Pay-per-character API | Open-source, self-hosted |
| Naturalness | Robotic even with SSML | Human-like by design |
| Flexibility | Rigid parameter ranges | Free-form descriptions |

Parler-TTS's description-based approach is architecturally superior — it controls the *character* of the voice, not just mechanical parameters.

---

### Why Modal

Both models (emotion classifier + Parler-TTS Large) run on **Modal serverless GPUs**:

- **No server management** — containers spin up on demand, scale to zero when idle
- **Model caching via Volumes** — weights are downloaded once and persisted, cold starts go from ~60s to ~5s
- **Single class pattern** — both models load in one `@modal.enter()` on an A10G, no cross-container calls
- **`@modal.fastapi_endpoint`** — one decorator exposes the class method as a public HTTPS endpoint

```python
@app.cls(gpu="A10G", volumes={CACHE_DIR: model_volume, OUTPUT_DIR: output_volume})
class EmpathyEngine:

    @modal.enter()
    def load_models(self):
        # loads both models once on container startup
        self.classifier = pipeline(...)   # emotion model
        self.tts_model  = ParlerTTSForConditionalGeneration.from_pretrained(...)

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        # full pipeline in one container, no remote calls
        result = self._run(request.text)
        return FileResponse(...)
```

---

## Project Structure

```
empathy_engine/
├── modal_app.py       # Modal app — EmpathyEngine class, GPU functions, FastAPI endpoint
├── voice_mapper.py    # Emotion → Parler-TTS description mapping (pure Python)
├── app.py             # Streamlit frontend
├── .env               # ENDPOINT_URL (not committed)
├── requirements.txt   # Local dependencies
└── README.md
```

---

## Environment Setup

### Prerequisites

- Python 3.11+
- [Modal account](https://modal.com) with CLI installed
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Tusharfv15/empathy-engine.git
cd empathy-engine
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install local dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
streamlit
requests
python-dotenv
modal
```

### 4. Set up Modal

```bash
pip install modal
modal setup   # opens browser for authentication
```

### 5. Create `.env` file

```bash
# .env
ENDPOINT_URL=https://<your-workspace>--empathy-engine-inference.modal.run
```

> You'll get this URL after deploying in the next step.

---

## Modal Deployment
<img width="1867" height="758" alt="image" src="https://github.com/user-attachments/assets/ae68e837-bd64-41fa-9a05-026777f2a28a" />

The entire backend (both models + API) is deployed as a single Modal app.

### How Modal deployment works

**Image build** — Modal builds a Docker-like image with all dependencies installed. This happens once and is cached:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "torch", "accelerate", "parler-tts", "soundfile", "fastapi[standard]")
    .add_local_python_source("voice_mapper")  # bundles local Python file into image
)
```

**Model caching via Volumes** — Modal Volumes persist data across container runs. Models are downloaded once and cached:

```python
model_volume  = modal.Volume.from_name("empathy-model-cache",  create_if_missing=True)
output_volume = modal.Volume.from_name("empathy-audio-output", create_if_missing=True)

@app.cls(
    gpu="A10G",
    volumes={
        CACHE_DIR:  model_volume,   # HuggingFace model weights cached here
        OUTPUT_DIR: output_volume,  # generated .wav files stored here
    }
)
```

**`@modal.enter()`** — runs once when a container starts, loads both models into memory:

```python
@modal.enter()
def load_models(self):
    os.environ["HF_HOME"]      = CACHE_DIR
    os.environ["HF_HUB_CACHE"] = CACHE_DIR

    # Emotion model (~300MB) — runs on same A10G
    self.classifier = pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0,
        model_kwargs={"cache_dir": CACHE_DIR},
    )

    # Parler-TTS Large (~5GB) — runs on A10G
    self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-large-v1",
        cache_dir=CACHE_DIR,
    ).to(self.device)
```

**`@modal.fastapi_endpoint`** — exposes the class method as a public HTTPS POST endpoint:

```python
@modal.fastapi_endpoint(method="POST")
def inference(self, request: InferenceRequest):
    result = self._run(request.text)
    return FileResponse(
        path=local_path,
        media_type="audio/wav",
        headers={
            "X-Emotion":    result["emotion"],
            "X-Intensity":  str(result["intensity"]),
            "X-Tier":       result["tier"],
            "X-All-Scores": json.dumps(result["all_scores"]),
        }
    )
```

### Deploy

```bash
# Deploy to Modal (creates persistent endpoint)
modal deploy modal_app.py
```

Output:
```
✓ Created objects.
├── 🔨 Created mount voice_mapper.py
├── 🔨 Created empathy-model-cache volume
├── 🔨 Created empathy-audio-output volume
└── 🔨 Created EmpathyEngine.inference => https://<workspace>--empathy-engine-inference.modal.run
```

Copy the endpoint URL into your `.env` file.

### Verify volumes (check model cache)

```bash
modal volume ls empathy-model-cache
```

### Serve in dev mode (live reload)

```bash
modal serve modal_app.py
```

---

## Running the Application

### Start the Streamlit frontend

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Test the API directly

```bash
curl -X POST https://<your-endpoint>/inference \
  -H "Content-Type: application/json" \
  -d '{"text": "This is the best news I have heard all year!"}' \
  --dump-header - \
  -o output.wav
```

Response headers:
```
X-Emotion: joy
X-Intensity: 0.9412
X-Tier: high
X-All-Scores: {"joy": 0.9412, "surprise": 0.03, "neutral": 0.02, ...}
X-Description: A very enthusiastic, high energy voice with a fast pace...
```

---

## API Reference

### POST /inference

**Request body:**
```json
{
  "text": "string — input text to synthesize"
}
```

**Response:**
- Content-Type: `audio/wav`
- Body: raw `.wav` audio bytes

**Response headers:**

| Header | Type | Description |
|---|---|---|
| `X-Emotion` | `string` | Dominant emotion label |
| `X-Intensity` | `float` | Confidence score (0.0 – 1.0) |
| `X-Tier` | `string` | Intensity tier: `low`, `mid`, or `high` |
| `X-All-Scores` | `JSON string` | Scores for all 7 emotion labels |
| `X-Description` | `string` | Parler-TTS voice description used |

**Error responses:**

| Code | Reason |
|---|---|
| `400` | Empty text input |
| `500` | Internal pipeline error |

---

## Bonus Features

| Feature | Status | Implementation |
|---|---|---|
| Granular Emotions | ✅ | 7-class emotion model (anger, disgust, fear, joy, neutral, sadness, surprise) |
| Intensity Scaling | ✅ | 3-tier system (low/mid/high) based on model confidence score |
| Web Interface | ✅ | Streamlit frontend with audio player + confidence score bars |
| SSML Integration | — | Replaced by Parler-TTS natural language descriptions (architecturally superior) |
