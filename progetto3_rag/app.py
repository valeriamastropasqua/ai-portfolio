import os
import re
import math
import anthropic
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="RAG Assistant · Valeria", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 300; margin-bottom: 0.2rem; }
    .main-title span { color: #3FCF8E; font-weight: 600; }
    .subtitle { color: #8B8A96; margin-bottom: 2rem; }
    .chunk-box {
        background: rgba(63,207,142,0.05);
        border-left: 3px solid #3FCF8E;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #8B8A96;
    }
    .source-badge {
        display: inline-block;
        background: rgba(63,207,142,0.1);
        border: 1px solid rgba(63,207,142,0.2);
        color: #3FCF8E;
        border-radius: 100px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">RAG <span>Assistant</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Carica i tuoi documenti e chatta con loro usando l\'AI</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configurazione")
    api_key = st.text_input("Anthropic API Key", type="password",
                            value=os.getenv("ANTHROPIC_API_KEY", ""), placeholder="sk-ant-...")
    st.markdown("---")
    st.markdown("### 📚 Come funziona RAG")
    st.markdown("""
    1. **Carica** PDF o testi
    2. **Indicizza** — divide in chunks e calcola TF-IDF
    3. **Chiedi** — trova i chunks più rilevanti e risponde
    
    L'AI risponde solo usando i tuoi documenti.
    """)
    st.markdown("---")
    if st.button("🗑️ Svuota documenti", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "vocab" not in st.session_state:
    st.session_state.vocab = {}
if "documenti" not in st.session_state:
    st.session_state.documenti = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def tokenizza(testo: str) -> list[str]:
    return re.findall(r'\b[a-zA-ZàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]{2,}\b', testo.lower())


def dividi_in_chunks(testo: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    parole = testo.split()
    chunks = []
    i = 0
    while i < len(parole):
        chunk = " ".join(parole[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def costruisci_tfidf(chunks: list[dict]) -> tuple[dict, list[dict]]:
    n = len(chunks)
    if n == 0:
        return {}, []

    vocab = {}
    df = {}

    tokenizzati = []
    for chunk in chunks:
        tokens = tokenizza(chunk["testo"])
        tokenizzati.append(tokens)
        parole_uniche = set(tokens)
        for parola in parole_uniche:
            if parola not in vocab:
                vocab[parola] = len(vocab)
            df[parola] = df.get(parola, 0) + 1

    vettori = []
    for tokens in tokenizzati:
        vec = {}
        tf_raw = {}
        for t in tokens:
            tf_raw[t] = tf_raw.get(t, 0) + 1

        for parola, count in tf_raw.items():
            tf = count / len(tokens) if tokens else 0
            idf = math.log(n / df.get(parola, 1))
            vec[vocab[parola]] = tf * idf

        norma = math.sqrt(sum(v**2 for v in vec.values()))
        if norma > 0:
            vec = {k: v/norma for k, v in vec.items()}
        vettori.append(vec)

    return vocab, vettori


def similarita_coseno(vec1: dict, vec2: dict) -> float:
    return sum(vec1.get(k, 0) * v for k, v in vec2.items())


def cerca_chunks(domanda: str, vocab: dict, vettori: list[dict],
                  chunks: list[dict], n: int = 4) -> list[dict]:
    if not vettori:
        return []

    tokens = tokenizza(domanda)
    vec_query = {}
    for t in tokens:
        if t in vocab:
            vec_query[vocab[t]] = vec_query.get(vocab[t], 0) + 1

    norma = math.sqrt(sum(v**2 for v in vec_query.values()))
    if norma > 0:
        vec_query = {k: v/norma for k, v in vec_query.items()}

    scores = []
    for i, vec in enumerate(vettori):
        sim = similarita_coseno(vec_query, vec)
        scores.append((sim, i))

    scores.sort(reverse=True)

    risultati = []
    for sim, idx in scores[:n]:
        if sim > 0:
            risultati.append({
                "testo": chunks[idx]["testo"],
                "fonte": chunks[idx]["fonte"],
                "similarita": round(sim * 100, 1)
            })

    return risultati


def aggiungi_documento(nome: str, testo: str):
    nuovi_chunks = dividi_in_chunks(testo)
    for chunk in nuovi_chunks:
        st.session_state.chunks.append({"testo": chunk, "fonte": nome})

    vocab, vettori = costruisci_tfidf(st.session_state.chunks)
    st.session_state.vocab = vocab
    st.session_state.tfidf = vettori

    return len(nuovi_chunks)


def genera_risposta(domanda: str, chunks_rilevanti: list[dict], api_key: str) -> str:
    contesto = "\n\n---\n\n".join([
        f"[Fonte: {c['fonte']}]\n{c['testo']}" for c in chunks_rilevanti
    ])

    prompt = f"""Sei un assistente che risponde alle domande basandosi ESCLUSIVAMENTE 
sui documenti forniti. Non inventare informazioni non presenti nel contesto.
Se la risposta non è nei documenti, dillo chiaramente.

DOCUMENTI DI RIFERIMENTO:
{contesto}

DOMANDA: {domanda}

Rispondi in italiano in modo chiaro e preciso. 
Cita la fonte quando possibile."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


col_sx, col_dx = st.columns([1, 2])

with col_sx:
    st.markdown("### 📁 Carica documenti")

    if st.button("📄 Usa documento di esempio", use_container_width=True):
        if "esempio" not in st.session_state.documenti:
            testo_esempio = """
INTELLIGENZA ARTIFICIALE E MACHINE LEARNING

L'intelligenza artificiale (IA) è un campo dell'informatica che crea sistemi 
capaci di eseguire compiti che richiedono intelligenza umana.

Il machine learning è una branca dell'IA che permette ai computer di imparare 
dall'esperienza. Gli algoritmi costruiscono modelli matematici basati su dati 
di addestramento senza essere esplicitamente programmati per ogni compito.

Il deep learning usa reti neurali con molti strati. È efficace per 
riconoscimento immagini, audio e linguaggio naturale.

RAG (Retrieval-Augmented Generation) combina ricerca di informazioni con 
generazione di testo. Recupera informazioni rilevanti da una knowledge base 
esterna per generare risposte più accurate e aggiornate.

I Large Language Models (LLM) come Claude di Anthropic e GPT di OpenAI sono 
addestrati su enormi quantità di testo. Possono generare testo, tradurre lingue, 
scrivere codice e rispondere a domande complesse.

APPLICAZIONI DELL'AI:
- Sanità: diagnosi mediche e scoperta di farmaci
- Finanza: rilevamento frodi e trading algoritmico
- Manifattura: controllo qualità e manutenzione predittiva
- Customer service: chatbot e assistenti virtuali

ETICA NELL'AI:
Lo sviluppo responsabile richiede attenzione a bias algoritmici, privacy dei dati,
trasparenza e impatto sociale. La sicurezza dei sistemi AI è fondamentale.
            """
            n = aggiungi_documento("Introduzione all'AI", testo_esempio)
            st.session_state.documenti.append("esempio")
            st.success(f"Documento aggiunto! ({n} chunks)")

    st.markdown("---")

    uploaded = st.file_uploader("Carica PDF o testo", type=["pdf", "txt"],
                                 accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.documenti:
                with st.spinner(f"Indicizzando {f.name}..."):
                    if f.name.endswith(".pdf"):
                        reader = PdfReader(f)
                        testo = "\n".join(p.extract_text() or "" for p in reader.pages)
                    else:
                        testo = f.read().decode("utf-8", errors="ignore")

                    if testo.strip():
                        n = aggiungi_documento(f.name, testo)
                        st.session_state.documenti.append(f.name)
                        st.success(f"✓ {f.name} ({n} chunks)")

    st.markdown("---")
    st.markdown("### 📊 Database")
    c1, c2 = st.columns(2)
    c1.metric("Chunks", len(st.session_state.chunks))
    c2.metric("Documenti", len(st.session_state.documenti))

with col_dx:
    st.markdown("### 💬 Chatta con i tuoi documenti")

    if not st.session_state.chunks:
        st.info("👈 Carica almeno un documento per iniziare.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "chunks" in msg:
                    with st.expander("📎 Fonti usate"):
                        for c in msg["chunks"]:
                            st.markdown(f'<span class="source-badge">📄 {c["fonte"]} — {c["similarita"]}% rilevante</span>',
                                       unsafe_allow_html=True)
                            st.markdown(f'<div class="chunk-box">{c["testo"][:300]}...</div>',
                                       unsafe_allow_html=True)

        domanda = st.chat_input("Fai una domanda sui tuoi documenti...")

        if domanda:
            if not api_key:
                st.error("Inserisci la API Key nella sidebar.")
            else:
                with st.chat_message("user"):
                    st.markdown(domanda)
                st.session_state.chat_history.append({"role": "user", "content": domanda})

                with st.chat_message("assistant"):
                    with st.spinner("Cercando nei documenti..."):
                        chunks_trovati = cerca_chunks(
                            domanda,
                            st.session_state.vocab,
                            st.session_state.tfidf,
                            st.session_state.chunks
                        )
                        if not chunks_trovati:
                            risposta = "Non ho trovato informazioni rilevanti nei documenti caricati."
                        else:
                            risposta = genera_risposta(domanda, chunks_trovati, api_key)

                    st.markdown(risposta)
                    with st.expander("📎 Fonti usate"):
                        for c in chunks_trovati:
                            st.markdown(f'<span class="source-badge">📄 {c["fonte"]} — {c["similarita"]}% rilevante</span>',
                                       unsafe_allow_html=True)
                            st.markdown(f'<div class="chunk-box">{c["testo"][:300]}...</div>',
                                       unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant", "content": risposta, "chunks": chunks_trovati
                })