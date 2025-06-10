# Q\&A with PDF using Groq LLM

This is a Streamlit-based web app that allows you to upload a PDF document and interact with it using natural language questions. It uses:

* **LangChain** for document processing and RAG (Retrieval-Augmented Generation)
* **Groq LLMs** (like `mixtral-8x7b-32768`) for fast and accurate answers
* **HuggingFace Sentence Transformers** for creating vector embeddings

---

## 🚀 Features

* Upload a PDF and ask questions about its content
* Extracts, chunks, and vectorizes text
* Answers are generated in real-time using Groq-hosted LLMs
* Improved UI with Streamlit widgets

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aryanajmera18/qa-with-pdf-groq
cd qa-with-pdf-groq
```

### 2. Create and Activate Virtual Environment

```bash
conda create -n genai python=3.10 -y
conda activate genai
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Keys Required

Create a `.env` file in the root folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🧠 Run the App

```bash
streamlit run app.py
```

Access the app at: [http://localhost:8501](http://localhost:8501)

---

## 📂 Folder Structure

```
.
├── app.py
├── requirements.txt
├── .env
├── README.md
```

---

## 🤖 LLMs Used

* `mixtral-8x7b-32768` via Groq (default)
* Other supported: `llama3-8b-8192`, `llama2-70b-4096`

---

## 🧪 To-Do / Improvements

* [ ] Add multi-file PDF support
* [ ] Add OpenAI/Anthropic fallback options
* [ ] Enable feedback loop for fine-tuning

---


---

## 📝 License

MIT License
