"""
╔══════════════════════════════════════════════════════════════╗
║  PROGETTO 3 — RAG su documenti propri                        ║
║  Portfolio AI Engineer · Valeria Mastropasqua                ║
╚══════════════════════════════════════════════════════════════╝

COSA IMPARI:
  - Cos'è RAG (Retrieval-Augmented Generation)
  - Come trasformare testi in vettori (embeddings)
  - Come usare un vector database (ChromaDB)
  - Come far rispondere l'AI solo con le info dei tuoi documenti

COME FUNZIONA RAG:
  1. Carica documenti (PDF o testo)
  2. Li divide in chunks (pezzetti)
  3. Converte ogni chunk in un vettore numerico (embedding)
  4. Salva i vettori in ChromaDB
  5. Quando fai una domanda:
     - La domanda viene convertita in vettore
     - ChromaDB trova i chunks più simili
     - Manda i chunks + domanda a Claude
     - Claude risponde basandosi SOLO su quei chunks

INSTALLAZIONE:
  pip install chromadb sentence-transformers pypdf anthropic streamlit

AVVIO:
  streamlit run app.py
"""

import os
import uuid
import anthropic
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ─── Configurazione pagina ────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant · Valeria",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 300; margin-bottom: 0.2rem; }
    .main-title span { color: #3FCF8E; font-weight: 600; }
    .subtitle { color: #8B8A96; margin-bottom: 2rem; }
    .chunk-box {
        background: #131316;
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

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurazione")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-..."
    )
    
    st.markdown("---")
    st.markdown("### 📚 Come funziona RAG")
    st.markdown("""
    1. **Carica** PDF o testi
    2. **Indicizza** — il sistema li divide e li vettorizza
    3. **Chiedi** — l'AI risponde usando solo i tuoi documenti
    
    L'AI non inventa: cita solo ciò che è nei documenti.
    """)
    
    st.markdown("---")
    st.markdown("### 🗑️ Reset")
    if st.button("Svuota tutti i documenti", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ─── Inizializzazione ChromaDB ────────────────────────────────
# ChromaDB è il nostro vector database locale.
# sentence-transformers converte testo in vettori numerici.
# Usiamo un modello leggero multilingua che funziona bene con l'italiano.

@st.cache_resource
def init_chroma():
    """Inizializza ChromaDB con il modello di embedding."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
        # Questo modello:
        # - È multilingua (funziona con italiano!)
        # - È leggero (~120MB)
        # - Si scarica automaticamente al primo avvio
    )
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="documenti_valeria",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}  # cosine similarity per confrontare vettori
    )
    return collection

collection = init_chroma()

# ─── Stato sessione ───────────────────────────────────────────
if "documenti_caricati" not in st.session_state:
    st.session_state.documenti_caricati = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Funzioni helper ──────────────────────────────────────────

def estrai_testo_pdf(file) -> str:
    """Estrae il testo da un PDF."""
    reader = PdfReader(file)
    testo = ""
    for pagina in reader.pages:
        testo += pagina.extract_text() + "\n"
    return testo

def dividi_in_chunks(testo: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Divide il testo in chunks sovrapposti.
    
    CONCETTO CHIAVE:
    - chunk_size: quante parole per chunk
    - overlap: quante parole si sovrappongono tra chunks adiacenti
    
    L'overlap serve per non perdere contesto ai bordi dei chunks.
    Es: "...fine chunk 1 | inizio chunk 2..." 
    Con overlap: "...fine chunk 1 overlap | overlap inizio chunk 2..."
    """
    parole = testo.split()
    chunks = []
    i = 0
    while i < len(parole):
        chunk = " ".join(parole[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def aggiungi_documento(nome: str, testo: str):
    """Aggiunge un documento al vector database."""
    chunks = dividi_in_chunks(testo)
    
    if not chunks:
        return 0
    
    # Ogni chunk ottiene un ID univoco e i metadati del documento originale
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadati = [{"fonte": nome, "chunk_index": i} for i, _ in enumerate(chunks)]
    
    # ChromaDB converte automaticamente i testi in vettori e li salva
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadati
    )
    
    return len(chunks)

def cerca_chunks_rilevanti(domanda: str, n_results: int = 4) -> list[dict]:
    """
    Cerca i chunks più rilevanti per la domanda.
    
    CONCETTO CHIAVE:
    ChromaDB converte la domanda in vettore e trova i chunks
    con la distanza coseno più bassa (= più simili semanticamente).
    Non è una ricerca per parole chiave — è una ricerca per significato!
    """
    risultati = collection.query(
        query_texts=[domanda],
        n_results=min(n_results, collection.count()),
    )
    
    chunks_trovati = []
    if risultati["documents"] and risultati["documents"][0]:
        for doc, meta, dist in zip(
            risultati["documents"][0],
            risultati["metadatas"][0],
            risultati["distances"][0]
        ):
            chunks_trovati.append({
                "testo": doc,
                "fonte": meta.get("fonte", "sconosciuta"),
                "similarita": round((1 - dist) * 100, 1)  # converti distanza in % similarità
            })
    
    return chunks_trovati

def genera_risposta(domanda: str, chunks: list[dict], api_key: str) -> str:
    """
    Genera la risposta usando Claude con i chunks come contesto.
    
    CONCETTO CHIAVE: il prompt RAG ha 3 parti:
    1. Istruzioni su come comportarsi
    2. Il contesto (chunks rilevanti)
    3. La domanda dell'utente
    """
    contesto = "\n\n---\n\n".join([
        f"[Fonte: {c['fonte']}]\n{c['testo']}" 
        for c in chunks
    ])
    
    prompt = f"""Sei un assistente che risponde alle domande basandosi ESCLUSIVAMENTE 
sui documenti forniti. Non inventare informazioni non presenti nel contesto.
Se la risposta non è nei documenti, dillo chiaramente.

DOCUMENTI DI RIFERIMENTO:
{contesto}

DOMANDA: {domanda}

Rispondi in italiano in modo chiaro e preciso. 
Cita la fonte quando possibile (es: "Secondo [nome documento]...")."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ─── Layout principale ────────────────────────────────────────
col_sinistra, col_destra = st.columns([1, 2])

# ── Colonna sinistra: caricamento documenti ───────────────────
with col_sinistra:
    st.markdown("### 📁 Carica documenti")
    
    # Documento di esempio integrato
    if st.button("📄 Usa documento di esempio", use_container_width=True):
        testo_esempio = """
        INTRODUZIONE ALL'INTELLIGENZA ARTIFICIALE
        
        L'intelligenza artificiale (IA) è un campo dell'informatica che si occupa 
        della creazione di sistemi in grado di eseguire compiti che normalmente 
        richiedono l'intelligenza umana.
        
        MACHINE LEARNING
        Il machine learning è una branca dell'IA che permette ai computer di 
        imparare dall'esperienza senza essere esplicitamente programmati. 
        Gli algoritmi di machine learning costruiscono modelli matematici basati 
        su dati di esempio chiamati dati di addestramento.
        
        DEEP LEARNING
        Il deep learning è un sottocampo del machine learning che utilizza reti 
        neurali artificiali con molti strati (layers). È particolarmente efficace 
        per il riconoscimento di immagini, il riconoscimento vocale e 
        l'elaborazione del linguaggio naturale.
        
        LARGE LANGUAGE MODELS (LLM)
        I Large Language Models sono modelli di deep learning addestrati su enormi 
        quantità di testo. Esempi famosi includono GPT di OpenAI e Claude di Anthropic.
        Questi modelli sono in grado di generare testo, tradurre lingue, scrivere 
        codice e rispondere a domande in modo coerente.
        
        RAG - RETRIEVAL AUGMENTED GENERATION
        RAG è una tecnica che combina la ricerca di informazioni con la generazione 
        di testo. Invece di affidarsi solo alla conoscenza del modello, RAG recupera 
        informazioni rilevanti da una base di conoscenza esterna e le usa come 
        contesto per generare risposte più accurate e aggiornate.
        
        APPLICAZIONI DELL'AI
        L'intelligenza artificiale trova applicazione in molti settori:
        - Sanità: diagnosi mediche e scoperta di farmaci
        - Finanza: rilevamento frodi e trading algoritmico  
        - Trasporti: veicoli autonomi e ottimizzazione dei percorsi
        - Manifattura: controllo qualità e manutenzione predittiva
        - Customer service: chatbot e assistenti virtuali
        
        ETICA NELL'AI
        Lo sviluppo responsabile dell'AI richiede attenzione a bias algoritmici,
        privacy dei dati, trasparenza dei modelli e impatto sociale. 
        Organizzazioni come Anthropic lavorano su AI sicura e benefica per l'umanità.
        """
        
        if "esempio_ai" not in st.session_state.documenti_caricati:
            n_chunks = aggiungi_documento("Introduzione all'AI", testo_esempio)
            st.session_state.documenti_caricati.append("esempio_ai")
            st.success(f"Documento aggiunto! ({n_chunks} chunks indicizzati)")
    
    st.markdown("---")
    
    # Upload file
    uploaded_files = st.file_uploader(
        "Oppure carica i tuoi file",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Puoi caricare più file contemporaneamente"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.documenti_caricati:
                with st.spinner(f"Indicizzando {file.name}..."):
                    if file.name.endswith(".pdf"):
                        testo = estrai_testo_pdf(file)
                    else:
                        testo = file.read().decode("utf-8", errors="ignore")
                    
                    if testo.strip():
                        n_chunks = aggiungi_documento(file.name, testo)
                        st.session_state.documenti_caricati.append(file.name)
                        st.success(f"✓ {file.name} ({n_chunks} chunks)")
                    else:
                        st.error(f"Nessun testo trovato in {file.name}")
    
    # Stato database
    st.markdown("---")
    st.markdown("### 📊 Database vettoriale")
    n_docs = collection.count()
    
    col_a, col_b = st.columns(2)
    col_a.metric("Chunks totali", n_docs)
    col_b.metric("Documenti", len(st.session_state.documenti_caricati))
    
    if st.session_state.documenti_caricati:
        st.markdown("**Documenti caricati:**")
        for doc in st.session_state.documenti_caricati:
            if doc != "esempio_ai":
                st.markdown(f"- 📄 {doc}")
            else:
                st.markdown("- 📄 Introduzione all'AI (esempio)")

# ── Colonna destra: chat ──────────────────────────────────────
with col_destra:
    st.markdown("### 💬 Chatta con i tuoi documenti")
    
    if collection.count() == 0:
        st.info("👈 Carica almeno un documento per iniziare a fare domande.")
    else:
        # Cronologia chat
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "chunks" in msg:
                    with st.expander("📎 Fonti usate per questa risposta"):
                        for chunk in msg["chunks"]:
                            st.markdown(f'<span class="source-badge">📄 {chunk["fonte"]} — {chunk["similarita"]}% rilevante</span>', 
                                       unsafe_allow_html=True)
                            st.markdown(f'<div class="chunk-box">{chunk["testo"][:300]}...</div>', 
                                       unsafe_allow_html=True)
        
        # Input domanda
        domanda = st.chat_input("Fai una domanda sui tuoi documenti...")
        
        if domanda:
            if not api_key:
                st.error("Inserisci la API Key nella sidebar.")
            else:
                # Mostra domanda utente
                with st.chat_message("user"):
                    st.markdown(domanda)
                st.session_state.chat_history.append({"role": "user", "content": domanda})
                
                # Cerca chunks rilevanti e genera risposta
                with st.chat_message("assistant"):
                    with st.spinner("Cercando nei documenti..."):
                        chunks = cerca_chunks_rilevanti(domanda)
                        
                        if not chunks:
                            risposta = "Non ho trovato informazioni rilevanti nei documenti caricati."
                        else:
                            risposta = genera_risposta(domanda, chunks, api_key)
                    
                    st.markdown(risposta)
                    
                    # Mostra le fonti usate
                    with st.expander("📎 Fonti usate per questa risposta"):
                        for chunk in chunks:
                            st.markdown(f'<span class="source-badge">📄 {chunk["fonte"]} — {chunk["similarita"]}% rilevante</span>', 
                                       unsafe_allow_html=True)
                            st.markdown(f'<div class="chunk-box">{chunk["testo"][:300]}...</div>', 
                                       unsafe_allow_html=True)
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": risposta,
                    "chunks": chunks
                })