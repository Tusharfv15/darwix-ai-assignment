"""
app.py — Streamlit frontend for the Empathy Engine
Run: streamlit run app.py
"""

import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
ENDPOINT_URL = os.getenv("ENDPOINT_URL")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Empathy Engine",
    page_icon="🎙️",
    layout="centered",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

.main-wrap {
    max-width: 720px;
    margin: 0 auto;
    padding: 60px 24px 80px;
}

.header { margin-bottom: 56px; }

.header-tag {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff6438;
    margin-bottom: 16px;
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 6vw, 60px);
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: -0.03em;
    margin-bottom: 16px;
}

.header-title span { color: #ff6438; }

.header-sub {
    font-size: 13px;
    color: #6b6760;
}

.input-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b6760;
    margin-bottom: 10px;
}

.result-card {
    background: #111118;
    border: 1px solid #1e1e2a;
    border-radius: 4px;
    padding: 28px;
    margin-bottom: 20px;
}

.score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.score-name {
    font-size: 15px;
    color: #6b6760;
    width: 80px;
}

.score-bar-bg {
    height: 2px;
}

.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

.score-val {
    font-size: 14px;
    width: 40px;
    text-align: right;
}

.desc-text {
    font-size: 17px;
    color: #4a4a58;
}
</style>
""", unsafe_allow_html=True)

# ── Emotion colors ─────────────────────────────────────────────────────────────
EMOTION_COLOR = {
    "joy": "#ffb830",
    "anger": "#ff3c3c",
    "sadness": "#5b8cff",
    "fear": "#a855f7",
    "surprise": "#ff6438",
    "disgust": "#22c55e",
    "neutral": "#6b6760",
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-wrap">
<div class="header">
    <div class="header-tag">◈ AI Voice Synthesis</div>
    <div class="header-title">Empathy<br><span>Engine</span></div>
    <div class="header-sub">
        Text → Emotion → Voice
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-label">Input text</div>',
            unsafe_allow_html=True)

text_input = st.text_area(
    label="text",
    placeholder="Type something with emotion...",
    height=140,
)

submit = st.button("Synthesize →", disabled=not bool(text_input.strip()))

# ── Synthesize ─────────────────────────────────────────────────────────────────
if submit and text_input.strip():
    if not ENDPOINT_URL:
        st.error("ENDPOINT_URL not set")
    else:
        with st.spinner("Synthesizing..."):
            try:
                response = requests.post(
                    ENDPOINT_URL,
                    json={"text": text_input.strip()},
                    timeout=300,
                )

                if response.status_code == 200:
                    emotion = response.headers.get("X-Emotion", "unknown")
                    intensity = float(response.headers.get("X-Intensity", "0"))
                    tier = response.headers.get("X-Tier", "—")
                    description = response.headers.get("X-Description", "")
                    all_scores = json.loads(
                        response.headers.get("X-All-Scores", "{}"))

                    color = EMOTION_COLOR.get(emotion, "#6b6760")

                    score_bars = ""
                    for label, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                        bar_color = color if label == emotion else "#2a2a38"
                        width = int(score * 100)

                        score_bars += (
                            f'<div class="score-row">'
                            f'<div class="score-name">{label}</div>'
                            f'<div class="score-bar-bg">'
                            f'<div class="score-bar-fill" style="width:{width}%; background:{bar_color};"></div>'
                            f'</div>'
                            f'<div class="score-val">{score:.2f}</div>'
                            f'</div>'
                        )

                    html_content = f"""<div class="result-card">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
<div style="padding:6px 14px;border-radius:3px;font-size:17px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:{color};border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03);">{emotion}</div>
<div style="font-size:16px;color:#6b6760;letter-spacing:0.1em;text-transform:uppercase;">{tier} intensity</div>
<div style="margin-left:auto;font-size:16px;color:#4a4a58;">{intensity:.4f}</div>
</div>

<div style="font-size:15px;letter-spacing:0.15em;text-transform:uppercase;color:#3a3a48;margin-bottom:12px;">Confidence scores</div>

{score_bars}

<div style="margin-top:18px;font-size:15px;letter-spacing:0.15em;text-transform:uppercase;color:#3a3a48;margin-bottom:8px;">Voice description</div>

<div style="font-size:17px;color:#4a4a58;line-height:1.6;border-left:2px solid #1e1e2a;padding-left:12px;">
{description}
</div>
</div>"""

                    st.markdown(html_content.strip(), unsafe_allow_html=True)

                    st.audio(response.content, format="audio/wav")

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(str(e))

st.markdown("</div>", unsafe_allow_html=True)