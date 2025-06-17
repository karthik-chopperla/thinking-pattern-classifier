# thinking_utils.py

import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def generate_questions(n=15):
    themes = ["stress", "decision making", "relationships", "goals", "failure", "self-reflection", "change"]
    templates = [
        "How do you usually respond when faced with {}?",
        "What is your first reaction to {}?",
        "Can you describe how you think about {}?",
        "When dealing with {}, what do you typically focus on?",
        "What emotions arise in you when you face {}?",
        "How do you prepare yourself mentally for {}?",
        "When thinking of {}, what thoughts dominate your mind?"
    ]
    questions = []
    for _ in range(n):
        t = random.choice(themes)
        temp = random.choice(templates)
        questions.append(temp.format(t))
    return questions

def get_mcq_choices(theme):
    choices = {
        "stress": ["Take a break", "Talk to someone", "Ignore it", "Overthink"],
        "goals": ["Set clear targets", "Work randomly", "Procrastinate", "Track progress"],
        "failure": ["Feel guilty", "Learn and move on", "Blame others", "Try again"],
        "relationships": ["Avoid conflict", "Confront directly", "Compromise", "Stay silent"],
        "decision making": ["Use logic", "Follow gut", "Ask friends", "Flip a coin"],
        "self-reflection": ["Journal daily", "Think deeply", "Never reflect", "Meditate"],
        "change": ["Embrace it", "Fear it", "Adapt slowly", "Avoid it"]
    }
    opts = choices.get(theme, [])
    return random.sample(opts, 3) if len(opts) >= 3 else opts

def detect_theme(question):
    themes = ["stress", "goals", "failure", "relationships", "decision making", "self-reflection", "change"]
    for theme in themes:
        if theme in question.lower():
            return theme
    return "stress"

def preprocess_texts(texts):
    cleaned = []
    for t in texts:
        tokens = word_tokenize(t.lower())
        filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
        cleaned.append(" ".join(filtered))
    return cleaned

def vectorize_texts(cleaned_texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_texts)
    return X, vectorizer

def cluster_thinking(X):
    best_k = 2
    best_score = -1
    for k in range(2, min(5, X.shape[0]) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels) if X.shape[0] > k else -1
        if score > best_score:
            best_score = score
            best_k = k
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    labels = final_model.fit_predict(X)
    return final_model, labels

def generate_wordcloud(texts):
    combined_text = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    buf = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return image_base64
