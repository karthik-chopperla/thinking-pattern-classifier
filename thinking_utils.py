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

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Step 1: Generate 20 open-ended psychological questions
def generate_questions():
    themes = {
        "stress": [
            "How do you typically respond when you're feeling stressed?",
            "What’s your first instinct when facing overwhelming pressure?",
            "Describe how you handle mental exhaustion."
        ],
        "goals": [
            "What motivates you to pursue long-term goals?",
            "How do you react when you fall behind on a personal objective?",
            "Describe your process for setting and sticking to goals."
        ],
        "failure": [
            "What’s your first thought after experiencing failure?",
            "How do you usually bounce back from a mistake?",
            "Do you reflect or move on quickly after things go wrong?"
        ],
        "relationships": [
            "How do you approach conflict in close relationships?",
            "What do you value most in communication with others?",
            "Describe a recent emotional interaction and how you handled it."
        ],
        "decision making": [
            "How do you make tough decisions?",
            "Do you prefer logic, emotion, or instinct when deciding?",
            "How do you handle decisions that affect others?"
        ],
        "self-reflection": [
            "How often do you reflect on your own thoughts?",
            "Do you notice patterns in your thinking over time?",
            "How do you evaluate your own behavior?"
        ],
        "change": [
            "How do you handle sudden life changes?",
            "Are you more comfortable with routine or spontaneity?",
            "What’s your mindset when things don’t go as planned?"
        ]
    }

    all_questions = []
    for theme, qs in themes.items():
        all_questions.extend(random.sample(qs, 2))  # Pick 2 from each theme

    random.shuffle(all_questions)
    return all_questions[:20]

# Step 2: Clean user responses
def preprocess_texts(texts):
    cleaned = []
    for t in texts:
        tokens = word_tokenize(t.lower())
        filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
        cleaned.append(" ".join(filtered))
    return cleaned

# Step 3: Convert text to vectors
def vectorize_texts(cleaned_texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_texts)
    return X, vectorizer

# Step 4: Cluster thinking styles
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

# Step 5: Visualize cluster keywords
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
