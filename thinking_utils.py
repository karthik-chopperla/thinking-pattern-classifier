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

# Step 1: Generate dynamic, open-ended questions
def generate_questions(num=10):
    themes = [
        "stress", "decision making", "relationships", "goals", "failure",
        "mental exhaustion", "thinking patterns", "personal growth", "habit change",
        "planning", "motivation", "emotional resilience", "conflict", "communication"
    ]
    templates = [
        "How do you handle {}?",
        "Describe your thoughts about {}.",
        "What do you usually do when facing {}?",
        "How do you reflect on {}?",
        "What do you value when thinking about {}?",
    ]
    questions = []
    sampled_themes = random.sample(themes, min(num, len(themes)))
    for t in sampled_themes:
        template = random.choice(templates)
        questions.append(template.format(t))
    return questions

# Step 2: Preprocess and vectorize text
def preprocess_texts(texts):
    cleaned = []
    for t in texts:
        t = t.lower()
        t = re.sub(r"[^a-z\s]", "", t)
        words = [w for w in t.split() if w not in {"the", "and", "is", "in", "to", "of", "a", "i"}]
        cleaned.append(" ".join(words))
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
