"""
modal_app.py — Empathy Engine
Single class, single A10G container.
Both models load on the same GPU — no cross-container remote calls.

Modes:
  CLI  : modal run modal_app.py --text "your text here"
  API  : modal serve modal_app.py  (dev)
         modal deploy modal_app.py (prod)
"""

import modal
from pydantic import BaseModel, Field

app = modal.App("empathy-engine")

# ── Volumes ────────────────────────────────────────────────────────────────────
model_volume  = modal.Volume.from_name("empathy-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("empathy-audio-output", create_if_missing=True)

# ── Image ──────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "parler-tts",
        "soundfile",
        "fastapi[standard]",
    )
    .add_local_python_source("voice_mapper")
)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
TTS_MODEL_ID     = "parler-tts/parler-tts-mini-v1"
CACHE_DIR        = "/model-cache"
OUTPUT_DIR       = "/audio-output"


# ── Pydantic request model ─────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    text: str = Field(description="Input text to synthesize")


# ══════════════════════════════════════════════════════════════════════════════
# EmpathyEngine — single class, full pipeline
# ══════════════════════════════════════════════════════════════════════════════
@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
    volumes={
        CACHE_DIR:  model_volume,
        OUTPUT_DIR: output_volume,
    },
)
class EmpathyEngine:

    @modal.enter()
    def load_models(self):
        import os
        import torch
        from transformers import pipeline
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        os.environ["HF_HOME"]      = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = CACHE_DIR

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[EmpathyEngine] Device: {self.device}")

        # ── Emotion model ──────────────────────────────────────────────────────
        print("[EmpathyEngine] Loading emotion model...")
        self.classifier = pipeline(
            task="text-classification",
            model=EMOTION_MODEL_ID,
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"cache_dir": CACHE_DIR},
        )
        print("[EmpathyEngine] Emotion model ready.")

        # ── Parler TTS model ───────────────────────────────────────────────────
        print("[EmpathyEngine] Loading Parler-TTS large...")
        self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            TTS_MODEL_ID,
            cache_dir=CACHE_DIR,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            TTS_MODEL_ID,
            cache_dir=CACHE_DIR,
        )
        print("[EmpathyEngine] Parler-TTS ready.")

    # ── Internal pipeline ──────────────────────────────────────────────────────
    def _detect_emotion(self, text: str) -> dict:
        raw        = self.classifier(text)[0]
        raw_sorted = sorted(raw, key=lambda x: x["score"], reverse=True)
        dominant   = raw_sorted[0]

        return {
            "emotion":    dominant["label"],
            "intensity":  round(dominant["score"], 4),
            "all_scores": {item["label"]: round(item["score"], 4) for item in raw_sorted},
        }

    def _synthesize(self, text: str, description: str) -> tuple[str, str]:
        import time
        import soundfile as sf

        input_ids        = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text,        return_tensors="pt").input_ids.to(self.device)

        generation = self.tts_model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
        )

        audio_np  = generation.cpu().numpy().squeeze()
        timestamp = int(time.time())
        filename  = f"output_{timestamp}.wav"
        filepath  = f"{OUTPUT_DIR}/{filename}"

        sf.write(filepath, audio_np, self.tts_model.config.sampling_rate)
        output_volume.commit()

        return filename, filepath

    def _run(self, text: str) -> dict:
        from voice_mapper import map_to_voice

        # Step 1 — Emotion detection
        print(f"[EmpathyEngine] Detecting emotion...")
        emotion_result = self._detect_emotion(text)
        print(f"[EmpathyEngine] {emotion_result['emotion']} ({emotion_result['intensity']})")

        # Step 2 — Voice mapping
        print(f"[EmpathyEngine] Mapping to voice description...")
        voice_result         = map_to_voice(emotion_result)
        voice_result["text"] = text
        print(f"[EmpathyEngine] Tier: {voice_result['tier']}")

        # Step 3 — TTS synthesis
        print(f"[EmpathyEngine] Synthesizing audio...")
        filename, filepath = self._synthesize(text, voice_result["description"])
        print(f"[EmpathyEngine] Saved → {filepath}")

        return {
            "text":        text,
            "emotion":     emotion_result["emotion"],
            "intensity":   emotion_result["intensity"],
            "all_scores":  emotion_result["all_scores"],
            "tier":        voice_result["tier"],
            "description": voice_result["description"],
            "filename":    filename,
        }

    # ── Modal method (for CLI) ─────────────────────────────────────────────────
    @modal.method()
    def run(self, text: str) -> dict:
        return self._run(text)

    # ── FastAPI endpoint ───────────────────────────────────────────────────────
    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        import json
        from fastapi import HTTPException
        from fastapi.responses import FileResponse

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        try:
            result = self._run(request.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Download wav locally from volume for FileResponse
        import os
        local_output = "/tmp/outputs"
        os.makedirs(local_output, exist_ok=True)
        local_path = f"{local_output}/{result['filename']}"

        with open(local_path, "wb") as f:
            for chunk in output_volume.read_file(result["filename"]):
                f.write(chunk)

        headers = {
            "X-Emotion":     result["emotion"],
            "X-Intensity":   str(result["intensity"]),
            "X-Tier":        result["tier"],
            "X-All-Scores":  json.dumps(result["all_scores"]),
            "X-Description": result["description"],
        }

        return FileResponse(
            path=local_path,
            media_type="audio/wav",
            filename=result["filename"],
            headers=headers,
        )


# ── Local entrypoint (CLI) ─────────────────────────────────────────────────────
@app.local_entrypoint()
def main(text: str = "This is the best news I've heard all year!"):
    import os

    engine       = EmpathyEngine()
    local_output = "./outputs"
    os.makedirs(local_output, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EMPATHY ENGINE")
    print("=" * 60)
    print(f"  Text: {text}")
    print("=" * 60)

    result     = engine.run.remote(text)
    local_path = f"{local_output}/{result['filename']}"

    with open(local_path, "wb") as f:
        for chunk in output_volume.read_file(result["filename"]):
            f.write(chunk)

    print(f"\nEmotion   : {result['emotion']} ({result['intensity']})")
    print(f"Tier      : {result['tier']}")
    print(f"All scores: {result['all_scores']}")
    print(f"Saved to  : {local_path}")
    print("=" * 60)