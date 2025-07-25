import streamlit as st
import os
import sys
import fitz
import requests
import json
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import re

# -------- Utility Functions --------

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def translate(text, target_lang, source_lang="auto"):
    try:
        if not text or target_lang == source_lang or target_lang is None:
            return text
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        return f"Translation failed: {e}"

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def generate_legal_answer(model, context, question, out_lang):
    prompt = f"""
You are a professional legal assistant. Answer the following question as clearly and concisely as possible in this language: {out_lang}.
Use numbered points and separate each answer by a new line.

Document:
{context}

Question:
{question}

Please provide your answer in {out_lang}.
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
            formatted_lines.append(f"**{line}**  ")
        else:
            formatted_lines.append(line + "  ")
    return "\n".join(formatted_lines)

def determine_better_answer(ans1, ans2):
    keywords = ["must", "shall", "liable", "verdict", "judgment", "convicted", "sentence", "acquitted"]
    score1 = sum(ans1.lower().count(k) for k in keywords) + len(ans1.split())
    score2 = sum(ans2.lower().count(k) for k in keywords) + len(ans2.split())
    return "LLaMA 3" if score1 >= score2 else "Gemma"

# ------- Language Command Recognition --------
lang_command_patterns = {
    "en": r"(in english|auf englisch|en anglais|en inglés|po angielsku)",
    "de": r"(in german|auf deutsch|en allemand|en alemán|po niemiecku)",
    "fr": r"(in french|auf französisch|en français|en francés|po francusku)",
    "es": r"(in spanish|auf spanisch|en espagnol|en español|po hiszpańsku)",
    "pl": r"(in polish|auf polnisch|en polonais|en polaco|po polsku)",
}

def extract_requested_language(prompt):
    prompt_lower = prompt.lower()
    for lang, pattern in lang_command_patterns.items():
        if re.search(pattern, prompt_lower):
            return lang
    return None

def get_language_display(lang_code):
    display = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pl": "Polish"
    }
    return display.get(lang_code, lang_code.upper())

# -------- Streamlit UI --------

st.set_page_config(page_title="Legal AI Assistant", layout="centered")
st.markdown("<h1 style='text-align: center;'>⚖️ LEGALEASE</h1><h3 style='text-align: center;'>Your Legal AI Assistant</h3>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded = st.file_uploader("📄 Upload a legal document (PDF only)", type=["pdf"])

if uploaded:
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", uploaded.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())
    raw_text = extract_text_from_pdf(temp_path)
    if not raw_text:
        st.error("No text found in the document.")
        st.stop()
    doc_lang = detect_language(raw_text[:1000])
    st.session_state['doc_lang'] = doc_lang
    st.session_state['text'] = raw_text
    st.session_state.messages = []
    st.success(f"✅ Document uploaded successfully. Detected language: **{get_language_display(doc_lang)}**")

st.sidebar.button("🔄 Reset chat", on_click=lambda: st.session_state.update(messages=[]))

# ---------------- Chat Logic -------------------

if "text" in st.session_state and "doc_lang" in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'], unsafe_allow_html=True)

    user_prompt = st.chat_input("Ask your legal question or specify a language (e.g. 'Please answer in French'):")

    if user_prompt and user_prompt.strip():
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        requested_lang = extract_requested_language(user_prompt)
        target_lang = requested_lang if requested_lang else st.session_state['doc_lang']
        target_lang_disp = get_language_display(target_lang)

        translated_prompt = (
            translate(user_prompt, "en", detect_language(user_prompt))
            if target_lang != "en" else user_prompt
        )

        context = st.session_state['text'][:4000]

        # Generate answers
        with st.spinner(f"🤖 LLaMA 3 is generating answer in {target_lang_disp}..."):
            ans_llama = generate_legal_answer("llama3", context, translated_prompt, target_lang_disp)
        with st.spinner(f"🤖 Gemma is generating answer in {target_lang_disp}..."):
            ans_gemma = generate_legal_answer("gemma", context, translated_prompt, target_lang_disp)

        # Enforce output language
        def enforce_lang(text, lang):
            if detect_language(text[:200]) != lang:
                return translate(text, lang, "en")
            return text

        result_llama = enforce_lang(ans_llama, target_lang)
        result_gemma = enforce_lang(ans_gemma, target_lang)

        score = calculate_reliability_score(ans_llama, ans_gemma)
        better = determine_better_answer(ans_llama, ans_gemma)

        assistant_reply = (
            f"#### 🦙 **LLaMA 3 Answer:**\n{format_response(result_llama)}\n\n"
            f"#### 🌐 **Gemma Answer:**\n{format_response(result_gemma)}\n\n"
            f"✅ **Reliability Score:** {score}%\n"
            f"🏆 **Final Judgment:** {better}'s answer is more informative and reliable."
        )

        with st.chat_message("assistant"):
            st.markdown(assistant_reply, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_reply
        })
