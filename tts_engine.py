"""
Module 3: TTS Engine
Model  : parler-tts/parler-tts-large-v1
Runtime: Modal A10G GPU
Input  : { description, emotion, intensity, tier }  ← output of Module 2
Output : .wav file saved to Modal Volume + local download
"""

import modal

# ── Modal image ────────────────────────────────────────────────────────────────
tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "parler-tts",
        "transformers",
        "torch",
        "accelerate",
        "soundfile",
    )
    .add_local_python_source("voice_mapper")
)

app = modal.App("empathy-engine-tts")

MODEL_ID   = "parler-tts/parler-tts-mini-v1"
CACHE_DIR  = "/model-cache"
OUTPUT_DIR = "/audio-output"

# ── Volumes ────────────────────────────────────────────────────────────────────
model_volume  = modal.Volume.from_name("tts-model-cache",  create_if_missing=True)
output_volume = modal.Volume.from_name("tts-audio-output", create_if_missing=True)


# ── Modal Class ────────────────────────────────────────────────────────────────
@app.cls(
    image=tts_image,
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
    volumes={
        CACHE_DIR:  model_volume,
        OUTPUT_DIR: output_volume,
    },
)
class TTSEngine:

    @modal.enter()
    def load_model(self):
        """
        Runs once when the container starts.
        Loads Parler-TTS large model and tokenizer into memory.
        """
        import os
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        os.environ["HF_HOME"]      = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = CACHE_DIR

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTSEngine] Loading model on {self.device}...")

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
        )

        print("[TTSEngine] Model loaded and ready.")

    @modal.method()
    def synthesize(self, voice_result: dict) -> str:
        """
        Synthesize speech from voice_result.

        Parameters
        ----------
        voice_result : dict
            Output from voice_mapper.map_to_voice()
            { text, description, emotion, intensity, tier }

        Returns
        -------
        str
            Filename of the saved .wav inside the output volume.
            e.g. "output_1711234567.wav"
        """
        import time
        import torch
        import soundfile as sf

        text        = voice_result["text"]
        description = voice_result["description"]
        emotion     = voice_result["emotion"]
        tier        = voice_result["tier"]

        print(f"[TTSEngine] Synthesizing — emotion: {emotion}, tier: {tier}")
        print(f"[TTSEngine] Description : {description[:80]}")
        print(f"[TTSEngine] Text        : {text[:80]}")

        # Tokenize description (how voice should sound) and text (what to say)
        description_tokens = self.tokenizer(
            description,
            return_tensors="pt",
        ).to(self.device)

        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        # Generate audio
        with torch.inference_mode():
            audio = self.model.generate(
                input_ids=description_tokens.input_ids,
                attention_mask=description_tokens.attention_mask,
                prompt_input_ids=text_tokens.input_ids,
                prompt_attention_mask=text_tokens.attention_mask,
            )

        # audio is a tensor of shape (1, num_samples) — squeeze to 1D numpy
        audio_np = audio.cpu().squeeze().numpy()

        # Save to output volume with timestamp filename
        timestamp = int(time.time())
        filename  = f"output_{timestamp}.wav"
        filepath  = f"{OUTPUT_DIR}/{filename}"

        sf.write(filepath, audio_np, self.model.config.sampling_rate)

        # Commit so the file is visible outside this container
        output_volume.commit()

        print(f"[TTSEngine] Saved → {filepath}")
        return filename


# ── Local entrypoint for quick testing ────────────────────────────────────────
@app.local_entrypoint()
def main():
    import os
    from voice_mapper import map_to_voice

    # Reference EmotionDetector across apps using from_name
    EmotionDetector = modal.Cls.from_name("empathy-engine-emotion", "EmotionDetector")

    test_cases = [
        "This is the best news I've heard all year! I'm so thrilled!",
        "I can't believe they cancelled my order again. This is so frustrating.",
        "The meeting is scheduled for 3pm tomorrow.",
    ]

    detector = EmotionDetector()
    tts      = TTSEngine()

    local_output = "./outputs"
    os.makedirs(local_output, exist_ok=True)

    print("\n" + "=" * 60)
    print("  FULL CHAIN TEST — Emotion → VoiceMap → TTS")
    print("=" * 60)

    for text in test_cases:
        # Module 1 — Modal T4
        emotion_result = detector.detect.remote(text)

        # Module 2 — local
        voice_result = map_to_voice(emotion_result)

        # Attach original text so TTS knows what to say
        voice_result["text"] = text

        # Module 3 — Modal A10G
        filename = tts.synthesize.remote(voice_result)

        # Download from Modal Volume to local ./outputs/
        local_path = f"{local_output}/{filename}"
        with open(local_path, "wb") as f:
            for chunk in output_volume.read_file(filename):
                f.write(chunk)

        print(f"\nText      : {text[:70]}")
        print(f"Emotion   : {emotion_result['emotion']} ({emotion_result['intensity']})")
        print(f"Tier      : {voice_result['tier']}")
        print(f"Saved to  : {local_path}")

    print("\n" + "=" * 60)