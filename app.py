import streamlit as st
import os
import sys
import fitz
import requests
import json
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------------- Utility Functions --------------------------------

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return f"Translation to English failed: {e}"

def translate_from_english(text, target_lang):
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        return f"Translation to target language failed: {e}"

def detect_language(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translator.translate(text)
        return translator._source_language  # undocumented but works
    except:
        return "en"

def generate_legal_answer(model, context, question):
    prompt = f"""
You are a professional legal assistant. Based on the document below, answer the following questions clearly and concisely. Use numbered points and separate each answer by a new line.

Document:
{context}

Questions:
1. Background of the case:
2. Key facts:
3. Final verdict/judgement:
4. Parties involved:
5. Criminal aspects and key takeaways:

Please provide your answers in the numbered format exactly as above.
"""

    for _ in range(2):  # Try twice if needed
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": model,
                    "prompt": prompt.strip(),
                    "stream": False
                })
            )
            result = response.json()
            output = result.get("response", "").strip()
            # Normalize spacing and line breaks
            output = output.replace("\r\n", "\n").replace("\r", "\n")
            if len(output.split()) > 5:
                return output
        except:
            continue
    return "⚠️ No valid response received."

def calculate_reliability_score(ans1, ans2):
    vectorizer = TfidfVectorizer().fit([ans1, ans2])
    vecs = vectorizer.transform([ans1, ans2])
    sim = cosine_similarity(vecs[0], vecs[1])[0][0]
    return round(sim * 100, 2)

def format_response(text):
    lines = text.strip().split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() and (line[1] == '.' or line[1] == ')')):
            # Numbered point - bold it
            formatted_lines.append(f"**{line}**  ")
        else:
            formatted_lines.append(line + "  ")
    return "\n".join(formatted_lines)

def determine_better_answer(ans1, ans2):
    keywords = ["must", "shall", "liable", "verdict", "judgment", "convicted", "sentence", "acquitted"]
    score1 = sum(ans1.lower().count(k) for k in keywords) + len(ans1.split())
    score2 = sum(ans2.lower().count(k) for k in keywords) + len(ans2.split())
    return "LLaMA 3" if score1 >= score2 else "Gemma"

# ---------------------------------------- Streamlit UI -----------------------------------------

st.set_page_config(page_title="Legal AI Assistant", layout="centered")
st.markdown("<h1 style='text-align: center;'>⚖️ LEGALEASE</h1><h3 style='text-align: center;'>Your Legal AI Assistant</h3>", unsafe_allow_html=True)

uploaded = st.file_uploader("📄 Upload a legal document (PDF only)", type=["pdf"])

query = st.text_area(
    "❓ Ask a legal question in any language",
    height=150,
    placeholder=(
        "Type your legal question here.\n"
    ),
)

if uploaded:
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", uploaded.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("🔍 Extracting text from document..."):
        raw_text = extract_text_from_pdf(temp_path)
        if not raw_text:
            st.error("No text found in the document.")
            st.stop()
        st.session_state["text"] = raw_text
    st.success("✅ Document uploaded and text extracted.")

if "text" in st.session_state:
    if query and query.strip():
        st.markdown("### 📌 Legal Answers")
        with st.spinner("🌍 Detecting language..."):
            source_lang = detect_language(query)
            translated_query = translate_to_english(query) if source_lang != "en" else query

        with st.spinner("🤖 Generating answers..."):
            context = st.session_state["text"][:4000]
            answer_llama = generate_legal_answer("llama3", context, translated_query)
            answer_gemma = generate_legal_answer("gemma", context, translated_query)

        # Translate answers back if needed
        if source_lang != "en":
            answer_llama_translated = translate_from_english(answer_llama, source_lang)
            answer_gemma_translated = translate_from_english(answer_gemma, source_lang)
        else:
            answer_llama_translated = answer_llama
            answer_gemma_translated = answer_gemma

        # Calculate similarity score
        similarity_score = calculate_reliability_score(answer_llama, answer_gemma)

        # Display answers nicely formatted
        st.markdown("#### 🦙 LLaMA 3 Answer:")
        st.markdown(format_response(answer_llama_translated), unsafe_allow_html=True)
        st.markdown("---")
        with st.expander("📘 Gemma's Comparison Answer"):
            st.markdown(format_response(answer_gemma_translated), unsafe_allow_html=True)

        st.markdown(f"### ✅ Reliability Score: **{similarity_score}%**")

        # Show only which answer is better
        better = determine_better_answer(answer_llama, answer_gemma)
        st.markdown(f"### 🏆 Final Judgment: **{better}'s answer is more informative and reliable.**")
