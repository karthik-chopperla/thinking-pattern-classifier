# Filename: app.py

import streamlit as st
from thinking_utils import (
    generate_questions,
    preprocess_texts,
    vectorize_texts,
    cluster_thinking,
    generate_wordcloud,
    get_top_keywords,
    generate_summary
)

st.set_page_config(page_title="Thinking Pattern Classifier", layout="centered")
st.title("🧠 Thinking Pattern Classifier")
st.write("Understand your thinking style based on how you respond to real-world mental patterns.")

mode = st.radio("How do you want to answer the questions?", ["Type your answers manually", "Select from MCQs"])

if "questions" not in st.session_state:
    st.session_state.questions = generate_questions()

responses = []

with st.form("response_form"):
    if mode == "Select from MCQs":
        mcq_options = {
            "stress": ["Overthink", "Ignore it", "Take a break"],
            "decision making": ["Plan ahead", "Go with gut", "Ask advice"],
            "relationships": ["Talk to someone", "Stay silent", "Ignore it"],
            "goals": ["Set clear targets", "Track progress", "Work randomly"],
            "failure": ["Feel guilty", "Blame others", "Learn and move on"],
            "mental exhaustion": ["Talk to someone", "Ignore it", "Take a break"],
            "habit change": ["Track daily", "Quit midway", "Reward yourself"],
            "conflict": ["Confront directly", "Compromise", "Stay silent"],
            "self-reflection": ["Talk to someone", "Ignore it", "Overthink"],
            "motivation": ["Set clear targets", "Reward yourself", "Ignore it"],
            "communication": ["Talk to someone", "Overthink", "Stay silent"],
            "fear": ["Embrace it", "Fear it", "Adapt slowly"],
            "planning": ["Plan ahead", "Go with gut", "Ignore it"],
            "change": ["Adapt slowly", "Embrace it", "Fear it"]
        }
        for i, q in enumerate(st.session_state.questions):
            theme = q.split(" ")[-1].strip("?.")
            options = mcq_options.get(theme.lower(), ["Option A", "Option B", "Option C"])
            answer = st.radio(f"Q{i+1}: {q}", options, key=f"q{i}")
            responses.append(answer)
    else:
        for i, q in enumerate(st.session_state.questions):
            answer = st.text_area(f"Q{i+1}: {q}", key=f"q{i}")
            responses.append(answer)
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
        st.image(f"data:image/png;base64,{image}", use_container_width=True)

    keywords = get_top_keywords(cluster_texts)
    summary = generate_summary(keywords)

    st.markdown("---")
    st.markdown("### 🧾 Summary Insight")
    st.success(summary)

    st.info("📝 Note: This is not a psychological diagnosis. It's a pattern-based insight generator.")
