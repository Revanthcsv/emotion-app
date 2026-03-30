import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import streamlit.components.v1 as components

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================
# Config & Constants
# =========================
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Emotion → Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Global Styling (Premium UX)
# =========================
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* Main Background */
.stApp {
  background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
  color: #ffffff;
  font-family: 'Inter', sans-serif;
}

/* Hide default header/footer */
header, footer {visibility: hidden;}

/* Custom Title Card */
.title-card {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 2rem;
  text-align: center;
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.title-card h1 {
  font-weight: 700;
  background: linear-gradient(90deg, #a8c0ff, #3f2b96);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
  color: white;
  border: none;
  border-radius: 50px;
  padding: 0.6rem 2rem;
  font-weight: 600;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.stButton>button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.3);
  background: linear-gradient(90deg, #5b7cd7 0%, #283858 100%);
}

/* Inputs */
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
  background-color: rgba(255, 255, 255, 0.05);
  color: white;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
  border-color: #4b6cb7;
  box-shadow: 0 0 0 1px #4b6cb7;
}

/* Cards */
.glass-card {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 1.5rem;
  margin-bottom: 1rem;
}

/* Song Card Fallback */
.song-card-fallback {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.8rem;
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: background 0.2s;
}
.song-card-fallback:hover {
  background: rgba(255, 255, 255, 0.08);
}
.song-info {
    display: flex;
    flex-direction: column;
}
.song-title {
    font-weight: 600;
    font-size: 1.05rem;
    color: #fff;
}
.song-artist {
    font-size: 0.9rem;
    color: #aaa;
}
a.spotify-btn {
    background: #1DB954;
    color: white;
    text-decoration: none;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
a.spotify-btn:hover {
    background: #1ed760;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0 0;
    color: #aaa;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #fff;
    border-bottom: 2px solid #4b6cb7;
}

/* Divider */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  margin: 1rem 0 1.3rem 0;
}

</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# =========================
# Header Section
# =========================
st.markdown(
    """
    <div class='title-card'>
        <h1>🎧 Emotion → Music Recommender</h1>
        <p style='color:#ddd; font-size:1.1rem; max-width: 700px; margin: 0 auto;'>
        Experience music that resonates with your soul. Detect your emotion from text or a photo, 
        and let us curate a playlist that <strong>matches</strong> or <strong>uplifts</strong> your mood.
        </p>
    </div>
    """, unsafe_allow_html=True
)

# =========================
# Session State
# =========================
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []
if "camera_enabled" not in st.session_state:
    st.session_state.camera_enabled = False

# =========================
# Helper Functions
# =========================
def get_spotify_embed_url(link: str):
    """Extracts Spotify Track ID and returns embed URL."""
    if not link:
        return None
    # Regex to find track ID from standard Spotify URLs
    match = re.search(r"track/([a-zA-Z0-9]+)", link)
    if match:
        track_id = match.group(1)
        return f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
    return None

def post_text_emotion(text: str):
    r = requests.post(f"{FASTAPI_URL}/predict/text", json={"text": text})
    r.raise_for_status()
    j = r.json()
    return j.get("mapped_emotion") or j.get("emotion") or j.get("detected_text_emotion")

def post_face_emotion_from_bytes(image_bytes: bytes):
    files = {"file": ("face.jpg", image_bytes, "image/jpeg")}
    r = requests.post(f"{FASTAPI_URL}/predict/face-image", files=files)
    r.raise_for_status()
    return r.json().get("emotion")

def post_recommendations(emotion: str, uplift: bool):
    payload = {"emotion": emotion, "uplift": uplift}
    r = requests.post(f"{FASTAPI_URL}/recommend", json=payload)
    r.raise_for_status()
    return r.json()

def render_recommendations(recs):
    if not recs:
        st.info("🎵 No recommendations yet. Try analyzing an emotion first!")
        return
    
    st.markdown("### 🎧 Your Curated Playlist")
    
    for item in recs:
        name = item.get("name", "Unknown Track")
        artist = item.get("artist", "Unknown Artist")
        link = item.get("link", "#")
        embed_url = get_spotify_embed_url(link)
        
        if embed_url:
            # Spotify Embed
            components.iframe(embed_url, height=80)
        else:
            # Fallback Card
            st.markdown(
                f"""
                <div class="song-card-fallback">
                    <div class="song-info">
                        <span class="song-title">{name}</span>
                        <span class="song-artist">{artist}</span>
                    </div>
                    <a class="spotify-btn" href="{link}" target="_blank">Play on Spotify</a>
                </div>
                """, unsafe_allow_html=True
            )

# =========================
# Main Layout & Tabs
# =========================
tab_face, tab_text, tab_logic = st.tabs([
    "📸 Face Detection",
    "✍️ Text Analysis",
    "🧠 How It Works"
])

# -------------------------
# Tab: Face (Upload / Camera)
# -------------------------
with tab_face:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### Capture Your Mood")
        st.markdown("Upload a photo or use your camera to let our AI read your facial expression.")
        
        capture_mode = st.radio(
            "Input Method",
            ["📁 Upload Image", "🎥 Use Camera"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        image_data = None
        
        if capture_mode == "📁 Upload Image":
            uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded:
                st.image(uploaded, caption="Your Image", use_container_width=True)
                image_data = uploaded.read()
                
        else: # Camera
            if not st.session_state.camera_enabled:
                if st.button("🔴 Activate Camera"):
                    st.session_state.camera_enabled = True
                    st.rerun()
            else:
                camera_image = st.camera_input("Snap a photo")
                if camera_image:
                    st.image(camera_image, caption="Captured Photo", use_container_width=True)
                    image_data = camera_image.getvalue()
                
                if st.button("❌ Turn Off Camera"):
                    st.session_state.camera_enabled = False
                    st.rerun()

    with col2:
        st.markdown("### Preferences")
        uplift_face = st.toggle("✨ Cheer me up! (Uplift Mode)", value=False, help="If enabled, we'll play songs to improve your mood instead of matching it.")
        
        st.markdown("---")
        
        if st.button("🚀 Analyze & Recommend", type="primary", use_container_width=True):
            if not image_data:
                st.warning("⚠️ Please provide an image first.")
            else:
                with st.spinner("🔍 Analyzing your expression..."):
                    try:
                        emo = post_face_emotion_from_bytes(image_data)
                        st.session_state.last_emotion = emo
                        
                        recs = post_recommendations(emo, uplift_face)
                        st.session_state.last_recs = recs
                        
                        st.success(f"Detected Emotion: **{emo.title()}**")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Results Area (in the same column for better flow)
        if st.session_state.last_emotion:
             render_recommendations(st.session_state.last_recs)


# -------------------------
# Tab: Text
# -------------------------
with tab_text:
    st.markdown(
        """
        <div class='glass-card'>
            <h3>💬 Tell us how you feel</h3>
            <p style='color:#ccc;'>Our BERT-based model will analyze your sentiment and find the perfect tracks.</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    col_t1, col_t2 = st.columns([2, 1], gap="large")
    
    with col_t1:
        user_text = st.text_area("Your thoughts...", height=150, placeholder="I'm feeling energetic and ready to take on the world!")
        
    with col_t2:
        st.markdown("### Options")
        uplift_text = st.toggle("✨ Cheer me up!", key="uplift_text", value=False)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("📝 Analyze Text", type="primary", use_container_width=True):
            if not user_text.strip():
                st.warning("⚠️ Please enter some text.")
            else:
                with st.spinner("🧠 Processing your words..."):
                    try:
                        emo = post_text_emotion(user_text)
                        st.session_state.last_emotion = emo
                        
                        recs = post_recommendations(emo, uplift_text)
                        st.session_state.last_recs = recs
                        
                        st.success(f"Detected Emotion: **{emo.title()}**")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Results below
    if st.session_state.last_emotion:
        st.markdown("---")
        render_recommendations(st.session_state.last_recs)


# -------------------------
# Tab: Logic (Technical)
# -------------------------
with tab_logic:
    st.markdown("## 🎼 How the Emotion → Music Recommender Works")

    st.markdown(
        """
        <div class='glass-card'>
        The system maps emotions onto a **Valence–Arousal (Energy)** space rather than simple labels.
        Songs are matched or uplifted based on their <strong>valence</strong> (pleasantness) and
        <strong>energy</strong> (arousal) values — features extracted from sources such as Spotify or the MUSE dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========== 1️⃣ COMPACT VALENCE–AROUSAL MAP + DESCRIPTION ==========
    col_plot, col_text = st.columns([1, 2], vertical_alignment="center")

    with col_plot:
        fig, ax = plt.subplots(figsize=(3.5, 3.3))
        ax.axhline(0.5, color='white', lw=1)
        ax.axvline(0.5, color='white', lw=1)

        # quadrants background
        ax.fill_betweenx([0.5, 1], 0.5, 1, color="#3CB371", alpha=0.25)   # Happy
        ax.fill_betweenx([0, 0.5], 0.5, 1, color="#FFD700", alpha=0.25)   # Relaxed
        ax.fill_betweenx([0.5, 1], 0, 0.5, color="#FF6B6B", alpha=0.25)   # Angry
        ax.fill_betweenx([0, 0.5], 0, 0.5, color="#6495ED", alpha=0.25)   # Sad

        # emotion points
        emotions = {
            "😡 Angry": (0.25, 0.85),
            "😃 Happy": (0.8, 0.75),
            "😢 Sad": (0.25, 0.25),
            "🙂 Relaxed": (0.75, 0.35),
            "😐 Neutral": (0.5, 0.5)
        }
        for label, (x, y) in emotions.items():
            ax.text(x, y, label, color='white', ha='center', va='center', fontsize=9, weight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Valence → Pleasantness", fontsize=9, color='white')
        ax.set_ylabel("Energy / Arousal ↑", fontsize=9, color='white')
        ax.set_facecolor("#1b153a")
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig, transparent=True)

    with col_text:
        st.markdown(
            """
            **Valence–Arousal Model**

            - **Valence** → how *pleasant* or *positive* the emotion feels  
            - **Arousal (Energy)** → how *activated* or *intense* the emotion is  

            Emotions are positioned in this 2-D space:
            - 😃 **Happy** → high valence, high arousal  
            - 😡 **Angry** → low valence, high arousal  
            - 😢 **Sad** → low valence, low arousal  
            - 🙂 **Relaxed** → high valence, low arousal  
            - 😐 **Neutral** → mid valence & energy  
            """
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== 2️⃣ CONGRUENT TABLE ==========
    st.markdown("### 🎯 Mood-Congruent Recommendation Logic")

    congruent = pd.DataFrame({
        "Emotion": ["Sad 😢", "Happy 😃", "Angry 😡", "Surprised 😲", "Neutral 😐"],
        "Valence": ["Low (<0.4)", "High (>0.6)", "Low (<0.4)", "Mid (0.4–0.7)", "Mid (0.4–0.6)"],
        "Energy": ["Low (<0.5)", "High (>0.5)", "High (>0.7)", "High (>0.6)", "Mid (0.4–0.6)"],
        "Psychological Basis": [
            "Sadness = unpleasant + calm (slow tempo, minor key)",
            "Happiness = pleasant + energetic (major key, upbeat rhythm)",
            "Anger = unpleasant + activated (fast, loud, aggressive tones)",
            "Surprise = mixed valence but high arousal (sudden shifts)",
            "Neutral = balanced affect (steady, smooth rhythm)"
        ]
    })
    st.dataframe(congruent, hide_index=True, width='stretch')

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== 3️⃣ UPLIFTING TABLE ==========
    st.markdown("### 🌤️ Mood-Uplifting Recommendation Logic")

    uplifting = pd.DataFrame({
        "Detected Emotion": ["Sad 😢", "Angry 😡", "Fearful 😱", "Happy 😃", "Neutral 😐"],
        "Target Valence": ["↑ (0.6–0.9)", "↑ (0.5–0.8)", "↑ (0.6–0.9)", "Maintain (0.6–1.0)", "↔ (0.4–0.6)"],
        "Target Energy": ["↑ / ↔ (0.4–0.7)", "↓ (0.3–0.6)", "↓ / ↔ (0.3–0.6)", "Maintain (0.5–1.0)", "↔ (0.4–0.6)"],
        "Psychological Purpose": [
            "Gently elevate mood with warm, hopeful tracks",
            "De-escalate tension with calm tones",
            "Provide comfort & reassurance",
            "Sustain positivity and engagement",
            "Encourage balanced exploration"
        ]
    })
    st.dataframe(uplifting, hide_index=True, width='stretch')

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== 4️⃣ TECH SUMMARY ==========
    st.markdown(
        """
        <div class='glass-card'>
        <h4>🔬 Technical Summary</h4>
        <p>
        The recommender converts your detected emotion into approximate <code>(valence, energy)</code> coordinates 
        and selects songs from the dataset that match these regions:
        </p>
        <ul>
            <li>🎯 <strong>Congruent mode</strong> → reinforces your current state (mood validation)</li>
            <li>🌤️ <strong>Uplifting mode</strong> → shifts toward higher valence or balanced energy (mood regulation)</li>
        </ul>
        <p>
        Thresholds are derived from affective computing literature and Spotify’s musical emotion metrics.  
        The approach balances empathy and regulation — validating emotions while promoting well-being.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
