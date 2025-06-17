# app.py

import streamlit as st
from thinking_utils import (
    generate_questions, preprocess_texts, vectorize_texts,
    cluster_thinking, generate_wordcloud
)

st.set_page_config(page_title="Thinking Pattern Classifier", layout="centered")
st.title("🧠 Thinking Pattern Classifier")
st.write("Understand your thinking style based on how you respond to real-world mental patterns.")

# Mode selection
mode = st.radio("How do you want to answer the questions?", ["Type your answers manually", "Select from MCQs"])

# Load or generate questions
if "questions" not in st.session_state:
    st.session_state.questions = generate_questions(num=15)

# Sample choices for MCQs
choices = {
    "stress": ["Overthink", "Ignore it", "Take a break"],
    "decision making": ["Talk to someone", "Go with gut", "Analyze deeply"],
    "relationships": ["Compromise", "Confront directly", "Stay silent"],
    "goals": ["Set clear targets", "Work randomly", "Track progress"],
    "failure": ["Feel guilty", "Blame others", "Learn and move on"],
    "mental exhaustion": ["Talk to someone", "Ignore it", "Take a break"],
    "thinking patterns": ["Overthink", "Take a break", "Talk to someone"],
    "personal growth": ["Set goals", "Avoid discomfort", "Reflect often"],
    "habit change": ["Track daily", "Quit midway", "Reward yourself"],
    "planning": ["Overplan", "Wing it", "Break into steps"],
    "motivation": ["Purpose", "Deadlines", "External praise"],
    "emotional resilience": ["Bounce back", "Cry", "Stay calm"],
    "conflict": ["Stay silent", "Argue", "Find middle ground"],
    "communication": ["Listen", "Interrupt", "Ignore"],
}

def get_mcq_choices(question):
    for theme, opts in choices.items():
        if theme in question.lower():
            return opts
    return ["Option A", "Option B", "Option C"]

responses = []

with st.form("thinking_form"):
    for i, question in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}: {question}**")
        if mode == "Select from MCQs":
            options = get_mcq_choices(question)
            response = st.radio("", options, key=f"q{i}")
        else:
            response = st.text_area("", key=f"q{i}")
        responses.append(response)
    submitted = st.form_submit_button("Analyze Thinking Pattern")

if submitted:
    cleaned = preprocess_texts(responses)
    X, vectorizer = vectorize_texts(cleaned)
    model, labels = cluster_thinking(X)
    cluster_texts = [[] for _ in range(model.n_clusters)]
    for idx, label in enumerate(labels):
        cluster_texts[label].append(cleaned[idx])

    st.success("🧠 Your cognitive styles have been clustered:")
    for i, texts in enumerate(cluster_texts):
        st.markdown(f"### 🧩 Cluster {i+1}")
        image = generate_wordcloud(texts)
        st.image(f"data:image/png;base64,{image}", use_container_width=True)  # ✅ FIXED HERE

        st.markdown("**Interpretation:** You tend to think with patterns involving keywords like:")
        keywords = list(set(" ".join(texts).split()))[:10]  # ✅ FIXED HERE
        st.code(", ".join(keywords))

    st.info("📝 Note: This is not a psychological diagnosis. It's a pattern-based insight generator.")
