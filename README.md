# ⚖️ Legal AI Assistant – "LegalEase"

LegalEase is a legal document question-answering assistant built using **Streamlit**, powered by **LLaMA 3** and **Gemma** open-source LLMs running via **Ollama**. It allows users to upload legal documents (PDFs), ask detailed legal questions in any language, and receive well-structured, side-by-side answers from two advanced AI models.

---

## 🚀 Features

- 📄 **PDF Upload**: Extracts full text from legal documents.
- 🧠 **Dual AI Responses**: Uses both `LLaMA 3` and `Gemma` to generate comparative legal insights.
- 🌐 **Multilingual Support**: Ask questions in **any language**, automatically translated and translated back.
- 📌 **Structured Legal Insights**:
  - Background of the case  
  - Key facts  
  - Final verdict/judgment  
  - Parties involved  
  - Criminal aspects and key takeaways  
- 🧾 **Formatted Output**: Neatly numbered, bolded responses with clean line breaks.
- ✅ **Reliability Score**: Calculates similarity between model responses using cosine similarity.
- 🏆 **Model Verdict**: Clearly shows which model provided the more reliable/legal-rich answer.

---

## 🛠 Installation Guide

### 1. Clone the Repository

```bash
git clone https://your-repo-url
cd legal-ai-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧠 Run Ollama Models

Make sure you have [Ollama](https://ollama.com) installed.

```bash
ollama serve
ollama pull llama3
ollama pull gemma
```

Then run both models:

```bash
ollama serve
ollama run llama3
ollama run gemma
```

---

## ▶ Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📝 Sample Questions

```text
What is the background of the case?
Who are the parties involved?
What was the final verdict?
What are the criminal aspects?
Key takeaways from this judgement?
```

Use **Shift+Enter** to format questions neatly.

---
