import os
import io
import base64
import requests
import streamlit as st

# =========================
# Config & Constants
# =========================
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Emotion → Music Recommender",
    page_icon="🎧",
    layout="wide"
)

# =========================
# Music-Themed Styling
# =========================
MUSIC_CSS = """
<style>
/* Background gradient */
.stApp {
  background: linear-gradient(135deg, #0f0c29 0%, #302b63 45%, #24243e 100%);
  color: #F2F5F7 !important;
}

/* Card look for sections */
.block-container {
  padding-top: 1.8rem;
}

.music-card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  backdrop-filter: blur(5px);
}

/* Headings */
h1, h2, h3, h4, h5 {
  color: #F6F7FB !important;
}

/* Buttons */
.stButton>button {
  background: #7c4dff;
  color: white;
  border-radius: 12px;
  border: 0;
  padding: 0.6rem 1rem;
}
.stButton>button:hover {
  background: #5f2eea;
}

/* Toggle label */
label[data-baseweb="toggle"] {
  font-weight: 600;
  color: #eaeaff;
}

/* Song item */
.song {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 0.7rem 0.9rem;
  border-radius: 12px;
  margin-bottom: 0.45rem;
}

/* Link style */
a.song-link {
  color: #9ad0ff !important;
  text-decoration: none;
}
a.song-link:hover {
  text-decoration: underline;
}

/* Subtle divider */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
  margin: 0.8rem 0 1rem 0;
}
</style>
"""
st.markdown(MUSIC_CSS, unsafe_allow_html=True)

# =========================
# Header
# =========================
col_logo, col_title = st.columns([1, 7])
with col_logo:
    st.markdown("### 🎶")
with col_title:
    st.markdown("## **Emotion → Music Recommender**")
    st.markdown(
        "<div class='music-card'>Tell me how you feel (or show me), "
        "and I’ll spin you a matching or uplifting playlist. 🎧✨</div>",
        unsafe_allow_html=True
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Keep state
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []

# =========================
# Helpers
# =========================
def post_text_emotion(text: str):
    url = f"{FASTAPI_URL}/predict/text"
    r = requests.post(url, json={"text": text}, timeout=60)
    r.raise_for_status()
    j = r.json()
    # prefer mapped emotion if available
    return j.get("mapped_emotion") or j.get("emotion") or j.get("detected_text_emotion")

def post_face_emotion_from_bytes(image_bytes: bytes):
    """Uses the image endpoint (works for both upload & camera capture)."""
    url = f"{FASTAPI_URL}/predict/face-image"
    files = {"file": ("face.jpg", image_bytes, "image/jpeg")}
    r = requests.post(url, files=files, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j.get("emotion")

def post_webcam_emotion():
    """Calls the backend webcam endpoint (works only if backend can access a local camera)."""
    url = f"{FASTAPI_URL}/predict/face-webcam"
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    j = r.json()
    return j.get("emotion")

def post_recommendations(emotion: str, uplift: bool):
    url = f"{FASTAPI_URL}/recommend"
    payload = {"emotion": emotion, "uplift": uplift}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()  # list of {name, artist, link}

def render_recommendations(recs: list):
    if not recs:
        st.info("No recommendations yet.")
        return
    for item in recs:
        name = item.get("name", "Unknown Track")
        artist = item.get("artist", "Unknown Artist")
        link = item.get("link", "")
        st.markdown(
            f"""<div class="song">🎵 <strong>{name}</strong> — <em>{artist}</em><br>
            <a class="song-link" href="{link}" target="_blank">Open track</a></div>""",
            unsafe_allow_html=True
        )

# =========================
# Tabs
# =========================
tab_face, tab_webcam, tab_text = st.tabs(
    ["🖼️ Face (Upload / Camera)", "📷 Webcam (Backend)", "💬 Text"]
)

# -------------------------
# Tab: Face (Upload / Camera using Streamlit)
# -------------------------
with tab_face:
    st.markdown("### 🖼️ Detect from Image or Camera")
    st.markdown(
        "<div class='music-card'>Upload a face image or use your camera. "
        "We’ll detect the emotion with your CNN model, then recommend music.</div>",
        unsafe_allow_html=True
    )

    # Split layout for upload vs camera
    c1, c2 = st.columns(2, vertical_alignment="center")

    with c1:
        uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)

    with c2:
        # Toggle webcam activation
        if "camera_enabled" not in st.session_state:
            st.session_state.camera_enabled = False

        if not st.session_state.camera_enabled:
            if st.button("🎥 Enable Camera"):
                st.session_state.camera_enabled = True
                st.rerun()
        else:
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                st.image(camera_image, caption="Captured from camera", use_column_width=True)
            if st.button("❌ Close Camera"):
                st.session_state.camera_enabled = False
                st.rerun()

    uplift = st.toggle("Cheer me up instead of matching my mood 🎶", value=False)

    if st.button("Analyze & Recommend (Face)"):
        try:
            img_bytes = None
            if uploaded is not None:
                img_bytes = uploaded.read()
            elif st.session_state.get("camera_enabled") and "camera_image" in locals() and camera_image is not None:
                img_bytes = camera_image.getvalue()

            if not img_bytes:
                st.warning("Please upload or capture an image first.")
            else:
                emo = post_face_emotion_from_bytes(img_bytes)
                st.session_state.last_emotion = emo
                st.success(f"Detected emotion: **{emo}**")

                recs = post_recommendations(emo, uplift)
                st.session_state.last_recs = recs

                st.markdown("### 🎧 Your Playlist")
                render_recommendations(recs)
        except requests.exceptions.ConnectionError:
            st.error("Could not reach FastAPI. Is it running?")
        except requests.HTTPError as e:
            st.error(f"Server error: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# -------------------------
# Tab: Webcam (Backend-side capture)
# -------------------------
with tab_webcam:
    st.markdown("### 📷 Backend Webcam (optional)")
    st.markdown(
        "<div class='music-card'>This calls the backend’s webcam endpoint. "
        "Use only if your FastAPI server has direct access to a webcam.</div>",
        unsafe_allow_html=True
    )
    uplift2 = st.toggle("Cheer me up instead of matching my mood 🎶 ", key="uplift2", value=False)

    if st.button("Analyze & Recommend (Backend Webcam)"):
        try:
            emo = post_webcam_emotion()
            st.session_state.last_emotion = emo
            st.success(f"Detected emotion: **{emo}**")

            recs = post_recommendations(emo, uplift2)
            st.session_state.last_recs = recs

            st.markdown("### 🎧 Your Playlist")
            render_recommendations(recs)
        except requests.exceptions.ConnectionError:
            st.error("Could not reach FastAPI. Is it running (with webcam access)?")
        except requests.HTTPError as e:
            st.error(f"Server error: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# -------------------------
# Tab: Text
# -------------------------
with tab_text:
    st.markdown("### 💬 Detect from Text")
    st.markdown(
        "<div class='music-card'>Describe your feelings in words. "
        "We’ll map your text emotion to the 5-class facial set, and recommend songs.</div>",
        unsafe_allow_html=True
    )
    user_text = st.text_area("How are you feeling today?", "I'm feeling great and energized!")
    uplift3 = st.toggle("Cheer me up instead of matching my mood 🎶  ", key="uplift3", value=False)

    if st.button("Analyze & Recommend (Text)"):
        try:
            emo = post_text_emotion(user_text)
            st.session_state.last_emotion = emo
            st.success(f"Detected emotion: **{emo}**")

            recs = post_recommendations(emo, uplift3)
            st.session_state.last_recs = recs

            st.markdown("### 🎧 Your Playlist")
            render_recommendations(recs)
        except requests.exceptions.ConnectionError:
            st.error("Could not reach FastAPI. Is it running?")
        except requests.HTTPError as e:
            st.error(f"Server error: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# =========================
# Footer / Last results
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
with st.expander("Last detected emotion & recommendations"):
    st.write("**Last emotion:**", st.session_state.last_emotion)
    render_recommendations(st.session_state.last_recs)
