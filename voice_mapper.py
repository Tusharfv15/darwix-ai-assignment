"""
Module 2: Voice Mapper
Input  : { emotion, intensity, all_scores }  ← output of Module 1
Output : { description, emotion, intensity, tier }
No GPU needed — pure Python logic.
"""

# ── Intensity tier thresholds ──────────────────────────────────────────────────
LOW  = 0.5
HIGH = 0.8


def _get_tier(intensity: float) -> str:
    if intensity < LOW:
        return "low"
    elif intensity < HIGH:
        return "mid"
    else:
        return "high"


# ── Emotion → description mapping ─────────────────────────────────────────────
# Each emotion has 3 tiers: low, mid, high
# Descriptions are written as Parler-TTS prompt style:
# natural language describing how the voice should sound

# Appended to every description as recommended by Parler-TTS
QUALITY_SUFFIX = "The recording is of very high quality, with the speaker's voice sounding clear and very close up."

VOICE_MAP = {
    "joy": {
        "low":  "A cheerful and friendly voice with a slightly upbeat tone at a moderate pace.",
        "mid":  "An enthusiastic and warm voice with an upbeat tone and a slightly faster pace.",
        "high": "A very enthusiastic, high energy voice with a fast pace, warm tone, and expressive delivery.",
    },
    "anger": {
        "low":  "A firm and direct voice with a slightly clipped and tense tone.",
        "mid":  "A sharp, tense voice with a fast pace and a hard, forceful delivery.",
        "high": "A very intense, aggressive voice with a rapid pace, loud delivery, and a harsh tone.",
    },
    "sadness": {
        "low":  "A calm voice with a slightly subdued and low tone at a slow pace.",
        "mid":  "A soft, slow voice with a heavy tone and a quiet, melancholic delivery.",
        "high": "A very slow, quiet voice with a heavy, sorrowful tone and a deeply subdued delivery.",
    },
    "fear": {
        "low":  "A slightly nervous voice with a hesitant tone and occasional pauses.",
        "mid":  "An anxious, hesitant voice with an unsteady pace and a tense, worried tone.",
        "high": "A very fearful, trembling voice with an erratic pace, whispered tone, and nervous delivery.",
    },
    "surprise": {
        "low":  "A mildly surprised voice with a slightly raised tone and an alert delivery.",
        "mid":  "An excited, wide-eyed voice with a fast pace and a bright, animated tone.",
        "high": "A very shocked voice with a rapid, breathless pace and a highly animated, expressive tone.",
    },
    "disgust": {
        "low":  "A slightly flat voice with a cold and detached tone at a measured pace.",
        "mid":  "A cold, dismissive voice with a slow pace and a dry, contemptuous tone.",
        "high": "A very cold, slow voice with a deeply contemptuous tone and a heavy, deliberate delivery.",
    },
    "neutral": {
        "low":  "A calm, clear voice at a moderate pace with a natural and composed delivery.",
        "mid":  "A calm, clear voice at a moderate pace with a natural and composed delivery.",
        "high": "A calm, clear voice at a moderate pace with a natural and composed delivery.",
    },
}


# ── Main mapping function ──────────────────────────────────────────────────────
def map_to_voice(emotion_result: dict) -> dict:
    """
    Maps emotion detection output to a Parler-TTS voice description.

    Parameters
    ----------
    emotion_result : dict
        Output from EmotionDetector.detect()
        { emotion, intensity, all_scores }

    Returns
    -------
    {
        "description" : str,   # Parler-TTS prompt
        "emotion"     : str,   # dominant emotion
        "intensity"   : float, # confidence score
        "tier"        : str    # low / mid / high
    }
    """
    emotion   = emotion_result["emotion"]
    intensity = emotion_result["intensity"]

    # Fallback to neutral if somehow an unknown emotion comes through
    if emotion not in VOICE_MAP:
        print(f"[VoiceMapper] Unknown emotion '{emotion}', falling back to neutral.")
        emotion = "neutral"

    tier        = _get_tier(intensity)
    description = f"{VOICE_MAP[emotion][tier]} {QUALITY_SUFFIX}"

    result = {
        "description": description,
        "emotion":     emotion,
        "intensity":   intensity,
        "tier":        tier,
    }

    print(f"[VoiceMapper] {emotion} ({tier}, {intensity}) → '{description[:60]}...'")
    return result


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {"emotion": "joy",     "intensity": 0.94, "all_scores": {}},
        {"emotion": "joy",     "intensity": 0.60, "all_scores": {}},
        {"emotion": "joy",     "intensity": 0.35, "all_scores": {}},
        {"emotion": "anger",   "intensity": 0.88, "all_scores": {}},
        {"emotion": "sadness", "intensity": 0.72, "all_scores": {}},
        {"emotion": "fear",    "intensity": 0.45, "all_scores": {}},
        {"emotion": "neutral", "intensity": 0.91, "all_scores": {}},
    ]

    print("\n" + "=" * 60)
    print("  VOICE MAPPER — TEST RUN")
    print("=" * 60)

    for case in test_cases:
        result = map_to_voice(case)
        print(f"\nEmotion    : {result['emotion']}")
        print(f"Intensity  : {result['intensity']}  →  tier: {result['tier']}")
        print(f"Description: {result['description']}")

    print("\n" + "=" * 60)