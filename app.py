# app.py

import streamlit as st
from thinking_utils import (
    generate_questions,
    get_mcq_choices,
    detect_theme,
    preprocess_texts,
    vectorize_texts,
    cluster_thinking,
    generate_wordcloud
)

st.set_page_config(page_title="Thinking Pattern Classifier", layout="centered")

st.title("🧠 Thinking Pattern Classifier")
st.markdown("Understand your thinking style based on how you respond to real-world mental patterns.")

# Mode selection
mode = st.radio("How do you want to answer the questions?", ["Type your answers manually", "Select from MCQs"])
st.session_state.answer_mode = "MCQ" if "MCQ" in mode else "TEXT"

# Generate questions
if "questions" not in st.session_state:
    st.session_state.questions = generate_questions(15)

responses = []

with st.form("response_form"):
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}: {q}**")
        if st.session_state.answer_mode == "MCQ":
            theme = detect_theme(q)
            options = get_mcq_choices(theme)
            if options:
                answer = st.radio("", options, key=f"mcq_{i}")
            else:
                answer = st.text_input("No choices available. Type your answer:", key=f"mcq_fallback_{i}")
        else:
            answer = st.text_area("", key=f"text_{i}")
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
        st.image(f"data:image/png;base64,{image}", use_column_width=True)
        st.markdown("**Interpretation:** You tend to think with patterns involving keywords like:")
        st.code(", ".join(set(" ".join(texts).split())[:10]))

    st.info("📝 Note: This is not a psychological diagnosis. It's a pattern-based insight generator.")
