"""
api.py — FastAPI endpoint for the Empathy Engine
POST /synthesize → streams .wav file with emotion metadata in response headers

Deploy : modal deploy api.py
Serve  : modal serve api.py  (live reload during dev)
"""

import modal
from modal_app import app, tts_image

# ── API image — extends tts_image with web deps ───────────────────────────────
api_image = (
    tts_image
    .pip_install("fastapi")
    .add_local_python_source("pipeline")
    .add_local_python_source("modal_app")
    .add_local_python_source("voice_mapper")
)


@app.function(
    image=api_image,
    timeout=300,
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST")
def synthesize(item: dict):
    """
    POST /synthesize
    Body : { "text": "your text here" }

    Returns
    -------
    .wav audio stream with emotion metadata in response headers:
        X-Emotion     : detected emotion label
        X-Intensity   : confidence score (0.0 – 1.0)
        X-Tier        : intensity tier (low / mid / high)
        X-All-Scores  : JSON string of all 7 emotion scores
        X-Description : Parler-TTS voice description used
    """
    import json
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from pipeline import run

    text = item.get("text", "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = run(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    headers = {
        "X-Emotion":     result["emotion"],
        "X-Intensity":   str(result["intensity"]),
        "X-Tier":        result["tier"],
        "X-All-Scores":  json.dumps(result["all_scores"]),
        "X-Description": result["description"],
    }

    return FileResponse(
        path=result["local_path"],
        media_type="audio/wav",
        filename=result["filename"],
        headers=headers,
    )