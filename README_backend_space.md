---
title: Emotion Music Backend
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Emotion Music Recommender — Backend API

FastAPI backend that provides:
- `/predict/text` — Emotion detection from text (BERT)
- `/predict/face-image` — Emotion detection from uploaded image (Keras MobileNetV2)
- `/recommend` — Music recommendations by emotion

Visit `/docs` for the interactive API documentation.
