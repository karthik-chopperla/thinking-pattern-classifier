# app.py

import streamlit as st
from thinking_utils import (
    generate_questions, preprocess_texts,
    vectorize_texts, cluster_thinking,
    generate_wordcloud
)

st.set_page_config(page_title="Thinking Pattern Classifier", layout="centered")
st.title("🧠 Thinking Pattern Classifier")
st.write("Understand your thinking style based on how you respond to real-world mental patterns.")

# Step 1: Select input mode
mode = st.radio("How do you want to answer the questions?", ["Type your answers manually", "Select from MCQs"])

# Step 2: Generate questions once per session
if "questions" not in st.session_state:
    st.session_state.questions = generate_questions(num_questions=15)

responses = []

# Step 3: Display form for input
with st.form("response_form"):
    st.write("### Please answer the following:")
    
    if mode == "Type your answers manually":
        for i, q in enumerate(st.session_state.questions):
            ans = st.text_area(f"Q{i+1}: {q}", key=f"manual_q{i}")
            responses.append(ans)
    else:
        choices = [
            ["Overthink", "Take a break", "Talk to someone"],
            ["Ignore it", "Set goals", "Write it down"],
            ["Blame others", "Reflect deeply", "Adapt"],
            ["Talk to someone", "Stay silent", "Compromise"],
            ["Track progress", "Work randomly", "Set targets"]
        ]
        for i, q in enumerate(st.session_state.questions):
            options = choices[i % len(choices)]  # Cycle through sample choices
            ans = st.radio(f"Q{i+1}: {q}", options, key=f"mcq_q{i}")
            responses.append(ans)

    submitted = st.form_submit_button("Analyze Thinking Pattern")

# Step 4: Analyze responses
if submitted:
    if any(not r.strip() for r in responses):
        st.error("Please answer all the questions.")
    else:
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
            st.image(f"data:image/png;base64,{image}", use_column_width=True)
            st.markdown("**Interpretation:** You tend to think with patterns involving:")
            st.code(", ".join(set(" ".join(texts).split())[:10]))

        st.info("📝 Note: This is not a psychological diagnosis. It's a pattern-based insight generator.")
