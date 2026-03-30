import streamlit as st
import streamlit.components.v1 as components

st.title("🎵 Spotify Song Preview Player")

songs = [
    {
        "name": "Mr. Brightside",
        "artist": "The Killers",
        "spotify_id": "09ZQ5TmUG8TSL56n0knqrj",
    },
    {
        "name": "Wonderwall",
        "artist": "Oasis",
        "spotify_id": "06UfBBDISthj1ZJAtX4xjj",
    },
]

for song in songs:
    st.subheader(f"{song['name']} — {song['artist']}")
    embed_url = f"https://open.spotify.com/embed/track/{song['spotify_id']}?utm_source=generator&theme=0"
    components.iframe(embed_url, height=80)