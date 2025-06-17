# thinking_utils.py

import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Custom minimal tokenizer (no nltk)
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Stopword list (minimal, no nltk)
STOPWORDS = set("""
a an the is are was were be being been have has had do does did but if or because as until while of at by for with about
against between into through during before after above below to from up down in out on off over under again further then once
here there when where why how all any both each few more most other some such no nor not only own same so than too very can will just
""".split())

# Step 1: Generate dynamic, open-ended questions
def generate_questions(num=15):
    themes = [
        "stress", "decision making", "relationships", "goals", "failure",
        "personal growth", "mental exhaustion", "habit change", "planning", "motivation",
        "emotional resilience", "conflict", "communication", "thinking patterns"
    ]
    templates = [
        "What do you feel when you encounter {}?",
        "How do you usually respond when faced with {}?",
        "Can you describe how you think about {}?",
        "When dealing with {}, what do you typically focus on?",
        "What is your first reaction to {}?"
    ]
    questions = [random.choice(templates).format(t) for t in random.sample(themes, num)]
    return questions

# Step 2: Preprocess and vectorize text
def preprocess_texts(texts):
    cleaned = []
    for t in texts:
        tokens = simple_tokenize(t)
        filtered = [w for w in tokens if w not in STOPWORDS]
        cleaned.append(" ".join(filtered))
    return cleaned

def vectorize_texts(cleaned_texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_texts)
    return X, vectorizer

# Step 3: Cluster thinking styles
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

# Step 4: Generate Word Cloud as base64 image
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
