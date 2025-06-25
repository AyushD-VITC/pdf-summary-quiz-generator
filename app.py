import streamlit as st
from transformers import pipeline
import fitz
import torch

st.set_page_config(page_title="PDF Summary & Quiz Generator", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>📄 PDF Summary & Quiz Generator</h1>
""", unsafe_allow_html=True)

if torch.cuda.is_available():
    device = 0
else:
    device = -1

# Sidebar 
with st.sidebar:
    st.title("Gen AI Project By:")
    st.markdown("**Student Name:** Ayush Dwivedi")
    st.markdown("**College ID:** 23BCE1539")
    st.markdown("**Email:** ayush.dwivedi2023@vitstudent.ac.in")
    st.markdown("---")
    st.title("Instructions:")
    st.info("1. Upload a PDF\n2. Summary automaticly generated\n3. Enter the no. of questions to generate then click generate\n4. Questions generated\n5. Download Summary+Questions from the download button.\n\n-To generate a summary for a new pdf reload the site.")

# Load models (cached)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_qg_model():
    return pipeline("text2text-generation", model="valhalla/t5-base-e2e-qg", device=device)

summarizer = load_summarizer()
quiz_gen = load_qg_model()

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload your PDF file", type=["pdf"])

# Initialize session state
if "final_summary" not in st.session_state:
    st.session_state.final_summary = ""
if "questions" not in st.session_state:
    st.session_state.questions = []

# Extract & summarize
if uploaded_file and not st.session_state.final_summary:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = "".join([page.get_text() for page in doc])

    with st.expander("🔍 Preview Extracted Text"):
        st.write(full_text[:2000] + "...")

    st.subheader("Summary")
    with st.spinner("Generating summary..."):
        def chunk_text(text, max_tokens=600):
            words = text.split()
            return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

        chunks = chunk_text(full_text)
        summary = ""
        for i, chunk in enumerate(chunks):
            try:
                result = summarizer(chunk, max_length=200, min_length=80, do_sample=False)[0]['summary_text']
                summary += result.strip() + " "
            except Exception as e:
                st.error(f"Error summarizing chunk #{i}: {e}")

        final_summary = summary.strip()
        if final_summary:
            st.session_state.final_summary = final_summary
            st.success(final_summary)
        else:
            st.error("Could not generate summary.")

# Step 2: Show stored summary if available
elif st.session_state.final_summary:
    st.subheader("Summary")
    st.success(st.session_state.final_summary)

# Step 3: Question generation input + trigger
if st.session_state.final_summary:
    num_questions = st.number_input(
        "Enter number of quiz questions to generate(max 5)", min_value=1, max_value=10, value=3
    )

    if st.button("Generate Quiz Questions"):
        with st.spinner("Generating questions..."):
            try:
                prompt = f"generate questions: {st.session_state.final_summary}"
                result = quiz_gen(prompt, max_length=512, num_return_sequences=1, do_sample=False)
                raw_text = result[0]['generated_text']
                raw_questions = [q.strip() for q in raw_text.split("<sep>") if q.strip() and "?" in q]

                # Deduplicate and limit by user input
                seen = set()
                questions = []
                for q in raw_questions:
                    if q not in seen:
                        questions.append(q)
                        seen.add(q)
                    if len(questions) == num_questions:
                        break

                if questions:
                    st.session_state.questions = questions
                    st.subheader("Quiz Questions")
                    for i, q in enumerate(questions, 1):
                        st.write(f"{i}. {q}")
                else:
                    st.warning("No valid questions generated.")
            except Exception as e:
                st.error(f"Error generating quiz: {e}")

# Step 4: Download Button
if st.session_state.final_summary and st.session_state.questions:
    download_text = f"Summary:\n{st.session_state.final_summary}\n\nQuiz Questions:\n"
    download_text += "\n".join([f"{i+1}. {q}" for i, q in enumerate(st.session_state.questions)])
    st.download_button("Download Summary + Questions", download_text, file_name="summary_quiz.txt")
