"""
Module 1: Emotion Detector
Model  : j-hartmann/emotion-english-distilroberta-base
Runtime: Modal T4 GPU
Output : { emotion, intensity, all_scores }
"""

import modal

# ── Modal image ────────────────────────────────────────────────────────────────
emotion_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
    )
)

app = modal.App("empathy-engine-emotion")

MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# ── Modal Class ────────────────────────────────────────────────────────────────
@app.cls(
    image=emotion_image,
    gpu="T4",
    timeout=120,
    volumes={"/model-cache": modal.Volume.from_name("emotion-model-cache", create_if_missing=True)},
)
class EmotionDetector:

    @modal.enter()
    def load_model(self):
        """
        Runs once when the container starts.
        Model is loaded into memory and reused across all .detect() calls.
        """
        import os
        import torch
        from transformers import pipeline

        CACHE_DIR = "/model-cache"

        # HF_HUB_CACHE is the correct env var for huggingface_hub >= 0.17
        # HF_HOME is the root — setting both covers all HF libs
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = CACHE_DIR

        device = 0 if torch.cuda.is_available() else -1

        print(f"[EmotionDetector] Loading model on {'GPU' if device == 0 else 'CPU'}...")

        self.classifier = pipeline(
            task="text-classification",
            model=MODEL_ID,
            top_k=None,
            device=device,
            model_kwargs={"cache_dir": CACHE_DIR},  # explicitly pass cache_dir to the model loader
        )

        print("[EmotionDetector] Model loaded and ready.")

    @modal.method()
    def detect(self, text: str) -> dict:
        """
        Classify the emotion in `text`.

        Returns
        -------
        {
            "emotion"    : str,   # dominant emotion label
            "intensity"  : float, # confidence score 0.0 – 1.0
            "all_scores" : dict   # {label: score} for all 7 emotions
        }
        """
        raw: list[dict] = self.classifier(text)[0]
        raw_sorted = sorted(raw, key=lambda x: x["score"], reverse=True)

        dominant = raw_sorted[0]
        all_scores = {item["label"]: round(item["score"], 4) for item in raw_sorted}

        result = {
            "emotion": dominant["label"],
            "intensity": round(dominant["score"], 4),
            "all_scores": all_scores,
        }

        print(f"[EmotionDetector] '{text[:60]}' → {result['emotion']} ({result['intensity']})")
        return result


# ── Local entrypoint for quick testing ────────────────────────────────────────
@app.local_entrypoint()
def main():
    from voice_mapper import map_to_voice

    test_cases = [
        "This is the best news I've heard all year! I'm so thrilled!",
        "I can't believe they cancelled my order again. This is so frustrating.",
        "The meeting is scheduled for 3pm tomorrow.",
        "I'm really scared about the results of the test.",
        "Wow, I never expected that to happen, what a surprise!",
    ]

    print("\n" + "=" * 60)
    print("  EMOTION + VOICE MAPPER — CHAIN TEST")
    print("=" * 60)

    detector = EmotionDetector()

    for text in test_cases:
        # Module 1 — runs on Modal T4
        emotion_result = detector.detect.remote(text)

        # Module 2 — runs locally
        voice_result = map_to_voice(emotion_result)

        print(f"\nText       : {text[:70]}")
        print(f"Emotion    : {emotion_result['emotion']}  ({emotion_result['intensity']})")
        print(f"Tier       : {voice_result['tier']}")
        print(f"Description: {voice_result['description']}")

    print("\n" + "=" * 60)