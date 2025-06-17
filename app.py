import streamlit as st
from thinking_utils import generate_questions, preprocess_texts, vectorize_texts, cluster_thinking, generate_wordcloud
import random
import base64

st.set_page_config(page_title="Thinking Pattern Classifier", layout="centered")

st.title("🧠 Thinking Pattern Classifier")
st.write("Understand your thinking style based on how you respond to real-world mental patterns.")

# Step 0: Answer style
mode = st.radio(
    "How do you want to answer the questions?",
    ["Type your answers manually", "Select from MCQs"],
    help="You can either write detailed responses or choose from dynamic multiple choice answers."
)

# Step 1: Generate questions + extract themes
if "questions" not in st.session_state:
    st.session_state.questions = generate_questions()

responses = []

# Dynamic MCQ choices per theme
def get_mcq_choices(theme):
    choices = {
        "stress": [
            "I avoid thinking about it", "I break problems into steps", "I get overwhelmed", "I distract myself"
        ],
        "goals": [
            "I set clear timelines", "I feel anxious about progress", "I wait until I feel motivated", "I act based on short-term wins"
        ],
        "relationships": [
            "I think logically before responding", "I prioritize emotions", "I avoid conflict", "I focus on fairness"
        ],
        "failure": [
            "I reflect and learn from it", "I criticize myself harshly", "I pretend it didn’t happen", "I seek external blame"
        ],
        "decision making": [
            "I rely on intuition", "I analyze pros and cons", "I ask others", "I go with the quickest option"
        ]
    }
    return random.sample(choices.get(theme, []), 3)

with st.form("response_form"):
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}: {q}**")

        # Extract theme keyword
        theme = next((t for t in ["stress", "goals", "relationships", "failure", "decision making"] if t in q.lower()), "general")

        if mode == "Type your answers manually":
            answer = st.text_area("Your answer:", key=f"text_q{i}")
        else:
            options = get_mcq_choices(theme)
            answer = st.selectbox("Choose the most accurate response:", options, key=f"mcq_q{i}")

        responses.append(answer)

    submitted = st.form_submit_button("Analyze Thinking Pattern")

# Step 2: Analyze & Show Clusters
if submitted:
    cleaned = preprocess_texts(responses)
    X, vectorizer = vectorize_texts(cleaned)
    model, labels = cluster_thinking(X)
    cluster_texts = [[] for _ in range(model.n_clusters)]
    for idx, label in enumerate(labels):
        cluster_texts[label].append(cleaned[idx])

    report = "🧠 Thinking Pattern Report\n\n"
    st.success("🧠 Your cognitive styles have been clustered:")

    for i, texts in enumerate(cluster_texts):
        st.markdown(f"### 🧩 Cluster {i+1}")
        image = generate_wordcloud(texts)
        st.image(f"data:image/png;base64,{image}", use_column_width=True)

        top_words = ", ".join(set(" ".join(texts).split())[:10])
        st.markdown("**Interpretation:**")
        st.code(top_words)

        # Build report string
        report += f"Cluster {i+1}:\n"
        report += f"- Keywords: {top_words}\n"
        report += f"- Example thoughts: {random.choice(texts)}\n\n"

    st.info("📝 This is a pattern-based reflection tool, not a psychological diagnosis.")

    # Step 3: Download button
    def get_download_link(text):
        b64 = base64.b64encode(text.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="thinking_pattern_report.txt">📥 Download Full Report</a>'

    st.markdown("---")
    st.markdown("### 📄 Your Personalized Report")
    st.markdown(get_download_link(report), unsafe_allow_html=True)
