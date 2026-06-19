import os
import streamlit as st
import anthropic

st.set_page_config(
    page_title="DataSci Tutor · Valeria AI",
    page_icon="🎓",
    layout="centered",
)

st.markdown("""
<style>
    .stApp { background-color: #0D0D0F; color: #F0EFF4; }
    .main-header { 
        text-align: center; 
        padding: 2rem 0 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 1.5rem;
    }
    .main-header h1 { 
        font-size: 1.8rem; 
        font-weight: 300;
        color: #F0EFF4;
    }
    .main-header h1 span { color: #7C6FF7; font-weight: 600; }
    .main-header p { color: #8B8A96; font-size: 0.9rem; margin-top: 0.5rem; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>DataSci <span>Tutor</span> AI</h1>
    <p>Il tuo assistente personale per Data Science e Machine Learning</p>
</div>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """Sei DataSci Tutor, un assistente AI esperto in Data Science, 
Machine Learning e Intelligenza Artificiale, creato da Valeria Mastropasqua 
come progetto portfolio.

Il tuo stile:
- Spieghi concetti complessi con esempi pratici e analogie chiare
- Usi Python nelle risposte di codice, con commenti esplicativi
- Sei incoraggiante e paziente, ideale per chi impara
- Rispondi sempre in italiano
- Quando mostri codice, spieghi ogni parte importante
- Suggerisci sempre risorse o prossimi passi per approfondire

Le tue aree di expertise:
- Machine Learning (scikit-learn, algoritmi, metriche)
- Deep Learning (reti neurali, PyTorch, TensorFlow)
- Analisi dati (pandas, numpy, matplotlib, seaborn)
- LLM e AI generativa (prompt engineering, RAG, agenti)
- Statistica applicata e probabilità
- Data Engineering (pipeline, ETL)

Se ti chiedono qualcosa fuori dal tuo dominio, reindirizza gentilmente 
verso argomenti di Data Science e AI."""


@st.cache_resource
def get_client(api_key: str):
    return anthropic.Anthropic(api_key=api_key)


with st.sidebar:
    st.markdown("### ⚙️ Configurazione")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Ottieni la tua key su console.anthropic.com",
        placeholder="sk-ant-..."
    )

    st.markdown("---")
    st.markdown("### 🎛️ Parametri")

    model = st.selectbox(
        "Modello",
        ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        help="Haiku è più veloce ed economico, Sonnet è più potente"
    )

    max_tokens = st.slider(
        "Lunghezza massima risposta",
        min_value=256,
        max_value=2048,
        value=1024,
        step=128,
    )

    st.markdown("---")
    st.markdown("### 💡 Domande di esempio")

    examples = [
        "Cos'è il gradient descent?",
        "Spiegami la differenza tra overfitting e underfitting",
        "Come funziona un transformer?",
        "Scrivimi un esempio di regressione logistica in Python",
        "Cosa sono i vettori di embedding?",
    ]

    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_message = ex

    st.markdown("---")
    if st.button("🗑️ Svuota chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "pending_message" in st.session_state:
    user_input = st.session_state.pop("pending_message")
else:
    user_input = st.chat_input("Chiedi qualcosa su Data Science o AI...")

if user_input:
    if not api_key:
        st.error("⚠️ Inserisci la tua Anthropic API Key nella sidebar per iniziare.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Sto pensando..."):
            client = get_client(api_key)

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=st.session_state.messages,
            )

            reply = response.content[0].text

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.expander("📊 Dettagli chiamata API"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Token input", response.usage.input_tokens)
        col2.metric("Token output", response.usage.output_tokens)
        col3.metric("Modello", model.split("-")[1])