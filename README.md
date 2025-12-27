# PDF Analyser Gemini Chatbot

A lightweight **PDF analyser and question-answering chatbot** built with **Streamlit**, **TF-IDF**, and **Google Gemini**.
The app reads a user-uploaded PDF, generates a short summary, and answers questions **strictly based on the PDF content**.

This project uses a **TF-IDF–based Retrieval-Augmented Generation (RAG)** approach to reduce hallucinations and keep answers grounded in the document.

---

## Features

* Upload and read PDF files
* Split PDF text into small chunks
* Retrieve the most relevant chunks using TF-IDF + cosine similarity
* Generate summaries and answers with Google Gemini
* Chat interface with conversation history
* Session state to avoid recomputation
* Fixed retrieval setting (`TOP_K = 2`) for clean and focused answers

---

## How It Works (TF-IDF RAG)

1. Extract text from the uploaded PDF
2. Clean and split text into small chunks
3. Convert chunks into TF-IDF vectors
4. Convert the user question into a TF-IDF vector
5. Find the **top 2 most relevant chunks** using cosine similarity
6. Send only those chunks to Gemini as context
7. Generate an answer **based only on the retrieved PDF content**

---

## Tech Stack

* Python
* Streamlit
* PyPDF
* scikit-learn (TF-IDF, cosine similarity)
* Google Gemini API
* python-dotenv

---

## 🖥 Usage

1. Upload a PDF from the sidebar
2. Wait for the summary to be generated
3. Ask questions about the PDF in the chat section
4. The chatbot answers **only using the PDF content**

---

## ⚙️ Default Configuration

```python
MODEL = "gemini-2.5-flash"
MAX_CHARS_PER_CHUNK = 100
TOP_K = 2
```

* `TOP_K = 2` is fixed by design
* This keeps answers short, relevant, and less noisy

---

## License

This project is intended for educational and personal use.
Feel free to fork and extend it.
