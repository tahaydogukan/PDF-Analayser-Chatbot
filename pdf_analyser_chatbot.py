import os
import re
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

load_dotenv()

# Model ve Paramtetre Seçimi 
MODEL = "gemini-2.5-flash"
MAX_CHARS_PER_CHUNK = 100
TOP_K = 2  # Bağlama eklenecek parça sayısı

# Sayfanın Tarayıcı Sekmesindeki Başlık
st.set_page_config(page_title="PDF Analyser Chatbot", layout="wide")


# PDF İşleme Fonksiyonları(Veri Çıkartma, Veri Temizleme, Veriyi Parçalama)

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, max_chars: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            back = text.rfind(" ", start, end)
            if back != -1 and back > start + int(max_chars * 0.6):
                end = back
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


# TF-IDF Retriever

def build_retriever(chunks):
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(chunks)
    return vectorizer, X

def retrieve(chunks, vectorizer, X, query, top_k=2):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).flatten()
    idx = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in idx]


# Gemini Client ve Yanıt Oluşturma

def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY bulunamadı.")
        st.stop()
    return genai.Client(api_key=api_key)

def gemini_generate(client, prompt: str) -> str:
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return getattr(resp, "text", None) or str(resp)


# Arayüz Tasarımı

st.title("PDF Reader Gemini Chatbot by Streamlit")

with st.sidebar:
    st.header("PDF Yükle")
    st.caption("API Key : `GOOGLE_API_KEY`")
    uploaded = st.file_uploader("PDF Seç", type=["pdf"])


# Session State

if "ready" not in st.session_state:
    st.session_state.ready = False
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "X" not in st.session_state:
    st.session_state.X = None
if "messages" not in st.session_state:
    st.session_state.messages = []

client = get_client()


# Sayfayı 2 Sütuna Bölme
col1, col2 = st.columns([1, 1], gap="large")


# Sol Taraf: Özet

with col1:
    st.subheader("1) PDF Özeti")

    if uploaded:
        raw = extract_text_from_pdf(uploaded)
        raw = clean_text(raw)

        if len(raw) < 50:
            st.error("PDF’ten yeterli metin çıkarılamadı.")
            st.stop()

        chunks = chunk_text(raw, MAX_CHARS_PER_CHUNK)
        vectorizer, X = build_retriever(chunks)

        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.X = X

        # Özet (1 kere üret)
        if not st.session_state.ready:
            with st.spinner("Özet hazırlanıyor..."):
                prompt = f"""
Aşağıdaki metin bir kitabın/PDF'in içeriğidir.
Türkçe, kısa ve anlaşılır bir özet yaz.
- Bir kaç madde
- En önemli kavramlar ve ana fikir

METİN:
{raw[:1000]}
"""
                st.session_state.summary = gemini_generate(client, prompt)
                st.session_state.ready = True

        st.success("PDF yüklendi, özet hazır.")
        st.text_area("Kısa Özet", st.session_state.summary, height=350)


    else:
        st.info("Sol menüden bir PDF yükle.")


# Sağ Taraf: Chat, Soru sorma

with col2:
    st.subheader("2) PDF'e Dayalı Chat")
    st.caption(f"Sorulara cevap verirken PDF içinden en alakalı {TOP_K} parçayı bağlama ekler (TF-IDF).")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("PDF hakkındaki sorunuzu yazınız.")

    if user_q:
        if not st.session_state.ready:
            st.error("Önce bir PDF yüklemelisin.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        hits = retrieve(
            st.session_state.chunks,
            st.session_state.vectorizer,
            st.session_state.X,
            user_q,
            top_k=TOP_K,
        )

        context_blocks = "\n\n".join(
            [f"[Parça {i+1} | skor={score:.3f}]\n{text}" for i, (text, score) in enumerate(hits)]
        )

        system_rules = """
Kurallar:
- SADECE verilen PDF bağlamına dayan.
- Bağlamda yoksa açıkça: "PDF’te bu bilgi geçmiyor" de.
- Kısa, net, madde madde yanıtla.
- Gerekirse ilgili parçayı referansla (Parça 1, Parça 2 gibi).
"""

        prompt = f"""{system_rules}

PDF ÖZETİ (kısa):
{st.session_state.summary}

PDF BAĞLAM PARÇALARI:
{context_blocks}

SORU:
{user_q}
"""

        with st.chat_message("assistant"):
            with st.spinner("Cevap hazırlanıyor..."):
                answer = gemini_generate(client, prompt)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
