import os
import io
import anthropic
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Data Analyst AI · Valeria",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 300; margin-bottom: 0.2rem; }
    .main-title span { color: #7C6FF7; font-weight: 600; }
    .subtitle { color: #8B8A96; margin-bottom: 2rem; }
    .insight-box {
        background: #131316;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Data Analyst <span>AI</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Carica un CSV e ottieni analisi e insight in linguaggio naturale</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configurazione")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-..."
    )
    st.markdown("---")
    st.markdown("### 📁 Dataset di esempio")
    st.markdown("Non hai un CSV? Usa uno di questi:")

    use_sample = st.selectbox(
        "Dataset di esempio",
        ["Nessuno — carico il mio", "Vendite mensili", "Studenti e voti", "Temperatura città"]
    )


def get_sample_df(name: str) -> pd.DataFrame:
    if name == "Vendite mensili":
        return pd.DataFrame({
            "mese": ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                     "Lug", "Ago", "Set", "Ott", "Nov", "Dic"],
            "vendite": [12400, 13200, 15800, 14100, 16900, 18200,
                        17400, 15600, 19800, 21200, 23400, 28900],
            "costi": [8200, 8800, 9400, 9100, 10200, 10800,
                      10400, 9600, 11200, 12100, 13200, 15400],
            "regione": ["Nord","Nord","Centro","Centro","Sud","Sud",
                        "Nord","Centro","Sud","Nord","Centro","Sud"],
        })
    elif name == "Studenti e voti":
        import numpy as np
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            "studente": [f"Studente_{i}" for i in range(1, n+1)],
            "matematica": np.random.randint(55, 100, n),
            "italiano": np.random.randint(60, 100, n),
            "scienze": np.random.randint(50, 100, n),
            "ore_studio_settimana": np.random.randint(5, 30, n),
            "classe": np.random.choice(["3A", "3B", "4A", "4B"], n),
        })
    elif name == "Temperatura città":
        return pd.DataFrame({
            "citta": ["Milano","Roma","Napoli","Torino","Palermo"] * 4,
            "mese": ["Gen"]*5 + ["Apr"]*5 + ["Lug"]*5 + ["Ott"]*5,
            "temp_media": [2,8,10,1,13, 14,17,19,13,22, 27,32,35,26,38, 15,20,22,14,25],
            "precipitazioni_mm": [60,70,85,55,40, 45,55,70,50,30, 25,15,20,30,10, 80,90,100,75,60],
        })
    return None


df = None

if use_sample != "Nessuno — carico il mio":
    df = get_sample_df(use_sample)
    st.info(f"Dataset di esempio caricato: **{use_sample}**")
else:
    uploaded = st.file_uploader(
        "Carica il tuo file CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="Max 200MB"
    )
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.success(f"File caricato: **{uploaded.name}**")
        except Exception as e:
            st.error(f"Errore nel caricamento: {e}")

if df is not None:
    tab1, tab2, tab3 = st.tabs(["📋 Dati", "📈 Grafici", "🤖 Analisi AI"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Righe", df.shape[0])
        col2.metric("Colonne", df.shape[1])
        col3.metric("Valori mancanti", df.isnull().sum().sum())
        col4.metric("Colonne numeriche", len(df.select_dtypes(include='number').columns))

        st.markdown("#### Anteprima")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("#### Statistiche descrittive")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if not numeric_cols:
            st.warning("Nessuna colonna numerica trovata per i grafici.")
        else:
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### Distribuzione")
                selected_col = st.selectbox("Colonna", numeric_cols, key="dist_col")
                fig = px.histogram(df, x=selected_col, nbins=20,
                                   color_discrete_sequence=["#7C6FF7"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                if len(numeric_cols) >= 2:
                    st.markdown("#### Correlazione")
                    x_col = st.selectbox("Asse X", numeric_cols, key="x_col")
                    y_col = st.selectbox("Asse Y", numeric_cols,
                                         index=min(1, len(numeric_cols)-1), key="y_col")
                    color_col = categorical_cols[0] if categorical_cols else None
                    fig2 = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                      color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=True)

            if len(numeric_cols) >= 3:
                st.markdown("#### Matrice di correlazione")
                corr = df[numeric_cols].corr()
                fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                                  color_continuous_scale="RdBu_r")
                fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown("#### Chiedi all'AI qualcosa sui tuoi dati")

        preset_questions = [
            "Analizza questo dataset e dimmi i 5 insight più importanti",
            "Ci sono anomalie o valori anomali nei dati?",
            "Quali correlazioni interessanti vedi tra le variabili?",
            "Cosa mi consiglieresti di approfondire con questi dati?",
            "Scrivi un breve report esecutivo su questi dati",
        ]

        selected_preset = st.selectbox(
            "Domande rapide",
            ["— Scegli una domanda —"] + preset_questions
        )

        user_question = st.text_area(
            "Oppure scrivi la tua domanda:",
            value=selected_preset if selected_preset != "— Scegli una domanda —" else "",
            height=80,
            placeholder="Es: Qual è il mese con più vendite? Ci sono trend stagionali?"
        )

        analyze_btn = st.button("🤖 Analizza con AI", type="primary", use_container_width=True)

        if analyze_btn:
            if not api_key:
                st.error("Inserisci la API Key nella sidebar.")
            elif not user_question.strip():
                st.warning("Scrivi una domanda prima di procedere.")
            else:
                buffer = io.StringIO()
                df.info(buf=buffer)
                df_info = buffer.getvalue()

                context = f"""Hai a disposizione un dataset con queste caratteristiche:

STRUTTURA:
{df_info}

STATISTICHE DESCRITTIVE:
{df.describe().to_string()}

PRIME 10 RIGHE:
{df.head(10).to_string()}

VALORI MANCANTI PER COLONNA:
{df.isnull().sum().to_string()}
"""

                prompt = f"""{context}

DOMANDA DELL'UTENTE:
{user_question}

Rispondi in italiano in modo chiaro e strutturato. 
Usa bullet points e sezioni quando appropriato.
Basa le tue osservazioni SOLO sui dati forniti, senza inventare."""

                with st.spinner("L'AI sta analizzando i dati..."):
                    client = anthropic.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1500,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    analysis = response.content[0].text

                st.markdown("#### Analisi AI")
                st.markdown(analysis)

                with st.expander("📊 Dettagli chiamata API"):
                    col1, col2 = st.columns(2)
                    col1.metric("Token input (dati + domanda)", response.usage.input_tokens)
                    col2.metric("Token output (risposta)", response.usage.output_tokens)

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #8B8A96;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
        <p style="font-size: 1.1rem;">Carica un CSV dalla sidebar oppure scegli un dataset di esempio per iniziare.</p>
    </div>
    """, unsafe_allow_html=True)