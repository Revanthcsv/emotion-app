from transformers import pipeline

# Emotion Mapping for Text Classifier
# Map 11 emotions to the 6 emotions from the facial model
EMOTION_MAPPING = {
    'joy': 'Happy',
    'love': 'Happy',
    'excitement': 'Happy',
    'amusement': 'Happy',
    'interest': 'Neutral',
    'satisfaction': 'Neutral',
    'calmness': 'Neutral',
    'sadness': 'Sad',
    'anger': 'Angry',
    'fear': 'Fearful',
    'disgust': 'Angry',
    'surprise': 'Surprised'
}

def load_text_model():
    """Loads the pre-trained BERT-Emotions-Classifier for text analysis."""
    classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")
    return classifier

# Load the model once
TEXT_CLASSIFIER = load_text_model()