"""
Module 4: Pipeline
Orchestrates all 3 modules in sequence:
  1. EmotionDetector  (Modal T4)
  2. VoiceMapper      (local)
  3. TTSEngine        (Modal A10G)

Usage:
    from pipeline import run
    result = run("This is the best news ever!")
"""

import os
from modal_app import EmotionDetector, TTSEngine, output_volume
from voice_mapper import map_to_voice

LOCAL_OUTPUT_DIR = "./outputs"


def run(text: str) -> dict:
    """
    Full pipeline: text → emotion → voice description → audio file.

    Parameters
    ----------
    text : str
        Raw input text to synthesize.

    Returns
    -------
    {
        "text"        : str,   # original input
        "emotion"     : str,   # detected emotion
        "intensity"   : float, # confidence score
        "all_scores"  : dict,  # all 7 emotion scores
        "tier"        : str,   # low / mid / high
        "description" : str,   # Parler-TTS voice description
        "filename"    : str,   # output_<timestamp>.wav
        "local_path"  : str,   # ./outputs/output_<timestamp>.wav
    }
    """
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    detector = EmotionDetector()
    tts      = TTSEngine()

    # ── Step 1: Emotion Detection (Modal T4) ──────────────────────────────────
    print(f"\n[Pipeline] Step 1 — Detecting emotion...")
    emotion_result = detector.detect.remote(text)
    print(f"[Pipeline] Emotion: {emotion_result['emotion']} ({emotion_result['intensity']})")

    # ── Step 2: Voice Mapping (local) ─────────────────────────────────────────
    print(f"[Pipeline] Step 2 — Mapping to voice description...")
    voice_result         = map_to_voice(emotion_result)
    voice_result["text"] = text
    print(f"[Pipeline] Tier: {voice_result['tier']}")
    print(f"[Pipeline] Description: {voice_result['description'][:80]}...")

    # ── Step 3: TTS Synthesis (Modal A10G) ────────────────────────────────────
    print(f"[Pipeline] Step 3 — Synthesizing audio...")
    filename = tts.synthesize.remote(voice_result)

    # ── Step 4: Download from Modal Volume ────────────────────────────────────
    print(f"[Pipeline] Step 4 — Downloading {filename}...")
    local_path = f"{LOCAL_OUTPUT_DIR}/{filename}"
    with open(local_path, "wb") as f:
        for chunk in output_volume.read_file(filename):
            f.write(chunk)

    print(f"[Pipeline] Done → {local_path}\n")

    return {
        "text":        text,
        "emotion":     emotion_result["emotion"],
        "intensity":   emotion_result["intensity"],
        "all_scores":  emotion_result["all_scores"],
        "tier":        voice_result["tier"],
        "description": voice_result["description"],
        "filename":    filename,
        "local_path":  local_path,
    }