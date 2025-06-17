import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Step 1: Generate dynamic, open-ended questions
def generate_questions():
    themes = ["stress", "decision making", "relationships", "goals", "failure"]
    templates = [
        "How do you usually respond when faced with {}?",
        "What is your first reaction to {}?",
        "Can you describe how you think about {}?",
        "When dealing with {}, what do you typically focus on?"
    ]
    questions = [random.choice(templates).format(t) for t in random.sample(themes, 3)]
    return questions

# Step 2: Preprocess and vectorize text
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
