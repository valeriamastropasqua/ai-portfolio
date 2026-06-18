"""
╔══════════════════════════════════════════════════════════════╗
║  PROGETTO 4 — AI Agent autonomo                              ║
║  Portfolio AI Engineer · Valeria Mastropasqua                ║
╚══════════════════════════════════════════════════════════════╝

COSA IMPARI:
  - Come funziona il "tool use" (function calling) degli LLM
  - Come costruire un agente autonomo che pianifica i propri passi
  - Come dare all'AI la capacità di cercare sul web, leggere pagine,
    fare calcoli e scrivere file
  - Il loop agente: pensa → usa tool → osserva → pensa → ...

COME FUNZIONA UN AGENT:
  1. L'utente dà un obiettivo ("Ricerca le ultime notizie su AI")
  2. Claude decide QUALE tool usare e con QUALI parametri
  3. Il tool viene eseguito e il risultato torna a Claude
  4. Claude decide se ha finito o se usare un altro tool
  5. Ripete finché l'obiettivo è raggiunto

INSTALLAZIONE:
  pip install anthropic streamlit duckduckgo-search requests beautifulsoup4

AVVIO:
  streamlit run app.py
"""

import os
import json
import math
import datetime
import requests
import anthropic
import streamlit as st
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ─── Configurazione pagina ────────────────────────────────────
st.set_page_config(
    page_title="AI Agent · Valeria",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 300; margin-bottom: 0.2rem; }
    .main-title span { color: #F5A623; font-weight: 600; }
    .subtitle { color: #8B8A96; margin-bottom: 2rem; }
    .tool-call {
        background: rgba(245,166,35,0.05);
        border: 1px solid rgba(245,166,35,0.2);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .tool-result {
        background: rgba(63,207,142,0.05);
        border: 1px solid rgba(63,207,142,0.15);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #8B8A96;
    }
    .thinking {
        background: rgba(124,111,247,0.05);
        border-left: 3px solid #7C6FF7;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #C4C2D4;
    }
    .step-label {
        font-family: monospace;
        font-size: 0.75rem;
        color: #F5A623;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">AI <span>Agent</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Un agente autonomo che pianifica, cerca e ragiona per raggiungere i tuoi obiettivi</p>', unsafe_allow_html=True)

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
    st.markdown("### 🔧 Tool disponibili")
    st.markdown("""
    L'agente può usare questi tool:
    
    - **🔍 Ricerca web** — cerca notizie e informazioni
    - **📄 Leggi pagina** — legge il contenuto di un URL
    - **🧮 Calcolatrice** — esegue calcoli matematici
    - **📅 Data/ora** — ottieni data e ora corrente
    - **💾 Salva nota** — salva testo in un file
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Obiettivi di esempio")
    examples = [
        "Cerca le ultime notizie sull'intelligenza artificiale e fai un riassunto",
        "Calcola quanto guadagna in un anno chi prende 2500€ al mese, tasse escluse",
        "Che giorno della settimana era il 15 marzo 2020?",
        "Cerca informazioni su LangChain e spiegami cos'è",
        "Quanti secondi ci sono in un anno? Mostrami il calcolo passo per passo",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_goal = ex
    
    st.markdown("---")
    max_steps = st.slider("Passi massimi agente", 3, 10, 6)

# ─── Definizione dei TOOL ─────────────────────────────────────
# CONCETTO CHIAVE: i tool sono descritti in JSON schema.
# Claude legge queste descrizioni e decide autonomamente
# quale tool chiamare e con quali parametri.

TOOLS = [
    {
        "name": "web_search",
        "description": "Cerca informazioni sul web usando DuckDuckGo. Usa questo tool per trovare notizie recenti, fatti, o qualsiasi informazione che non conosci già.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La query di ricerca in italiano o inglese"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Numero massimo di risultati (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_webpage",
        "description": "Legge il contenuto testuale di una pagina web dato il suo URL. Usa dopo web_search per approfondire un risultato specifico.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "L'URL completo della pagina da leggere"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "calculate",
        "description": "Esegue calcoli matematici. Usa espressioni Python valide. Supporta: +, -, *, /, **, math.sqrt(), math.pi, ecc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Espressione matematica Python da valutare. Es: '2500 * 12', 'math.sqrt(144)', '(100 * 0.23)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_datetime",
        "description": "Restituisce la data e l'ora corrente, oppure informazioni su una data specifica.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_string": {
                    "type": "string",
                    "description": "Data opzionale nel formato YYYY-MM-DD per ottenere info su quella data. Lascia vuoto per la data corrente.",
                    "default": ""
                }
            }
        }
    },
    {
        "name": "save_note",
        "description": "Salva del testo in un file di testo locale. Utile per salvare riassunti, risultati o note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Nome del file (senza estensione). Es: 'riassunto_ai_news'"
                },
                "content": {
                    "type": "string",
                    "description": "Il contenuto da salvare nel file"
                }
            },
            "required": ["filename", "content"]
        }
    }
]

# ─── Implementazione dei TOOL ─────────────────────────────────

def web_search(query: str, max_results: int = 5) -> str:
    """Esegue una ricerca web con DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            risultati = list(ddgs.text(query, max_results=max_results))
        
        if not risultati:
            return "Nessun risultato trovato."
        
        output = f"Risultati per '{query}':\n\n"
        for i, r in enumerate(risultati, 1):
            output += f"{i}. **{r.get('title', 'N/A')}**\n"
            output += f"   URL: {r.get('href', 'N/A')}\n"
            output += f"   {r.get('body', 'N/A')}\n\n"
        
        return output
    except Exception as e:
        return f"Errore nella ricerca: {str(e)}"

def read_webpage(url: str) -> str:
    """Legge il contenuto testuale di una pagina web."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Agent/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Rimuovi script, style e nav
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        testo = soup.get_text(separator="\n", strip=True)
        
        # Limita a 3000 caratteri per non sovraccaricare il contesto
        if len(testo) > 3000:
            testo = testo[:3000] + "\n\n[...contenuto troncato per lunghezza...]"
        
        return f"Contenuto di {url}:\n\n{testo}"
    except Exception as e:
        return f"Errore nella lettura della pagina: {str(e)}"

def calculate(expression: str) -> str:
    """Esegue calcoli matematici in modo sicuro."""
    try:
        # Namespace sicuro: solo funzioni math permesse
        safe_namespace = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }
        risultato = eval(expression, {"__builtins__": {}}, safe_namespace)
        return f"Risultato di `{expression}` = **{risultato}**"
    except Exception as e:
        return f"Errore nel calcolo: {str(e)}"

def get_datetime(date_string: str = "") -> str:
    """Restituisce informazioni sulla data corrente o su una data specifica."""
    giorni = ["lunedì","martedì","mercoledì","giovedì","venerdì","sabato","domenica"]
    mesi = ["","gennaio","febbraio","marzo","aprile","maggio","giugno",
            "luglio","agosto","settembre","ottobre","novembre","dicembre"]
    
    if date_string:
        try:
            data = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            giorno_settimana = giorni[data.weekday()]
            return (f"La data {date_string} era un **{giorno_settimana}**, "
                    f"{data.day} {mesi[data.month]} {data.year}.")
        except:
            return f"Formato data non valido. Usa YYYY-MM-DD."
    else:
        ora = datetime.datetime.now()
        giorno_settimana = giorni[ora.weekday()]
        return (f"Oggi è **{giorno_settimana}**, {ora.day} {mesi[ora.month]} {ora.year}. "
                f"Ore: {ora.strftime('%H:%M:%S')}")

def save_note(filename: str, content: str) -> str:
    """Salva contenuto in un file di testo."""
    try:
        filepath = f"{filename}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File salvato con successo: **{filepath}** ({len(content)} caratteri)"
    except Exception as e:
        return f"Errore nel salvataggio: {str(e)}"

def esegui_tool(nome: str, parametri: dict) -> str:
    """Router: esegue il tool richiesto da Claude."""
    if nome == "web_search":
        return web_search(**parametri)
    elif nome == "read_webpage":
        return read_webpage(**parametri)
    elif nome == "calculate":
        return calculate(**parametri)
    elif nome == "get_datetime":
        return get_datetime(**parametri)
    elif nome == "save_note":
        return save_note(**parametri)
    else:
        return f"Tool '{nome}' non riconosciuto."

# ─── Loop principale dell'agente ──────────────────────────────

def esegui_agente(obiettivo: str, api_key: str, max_steps: int, log_container):
    """
    Il cuore dell'agente: loop pensa → agisce → osserva.
    
    CONCETTO CHIAVE — il loop agente:
    1. Manda obiettivo + storia a Claude
    2. Claude risponde con testo O con una richiesta di tool
    3. Se tool: eseguiamo il tool, aggiungiamo il risultato alla storia
    4. Se testo finale: abbiamo finito
    5. Ripetiamo fino a max_steps
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # La storia è una lista di messaggi — come nella chat normale
    # ma con in più i messaggi di tipo "tool_use" e "tool_result"
    messages = [{"role": "user", "content": obiettivo}]
    
    system_prompt = """Sei un agente AI autonomo e metodico. 
Quando ricevi un obiettivo:
1. Pianifica i passi necessari
2. Usa i tool disponibili per raccogliere informazioni o eseguire azioni
3. Ragiona sui risultati ottenuti
4. Produci una risposta finale chiara e completa in italiano

Usa i tool in modo efficiente. Se hai già le informazioni necessarie, non cercare ancora.
Rispondi sempre in italiano."""
    
    risposta_finale = ""
    
    for step in range(max_steps):
        with log_container:
            st.markdown(f'<div class="step-label">→ Step {step + 1}</div>', 
                       unsafe_allow_html=True)
        
        # Chiamata a Claude con i tool disponibili
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        
        # Analizza la risposta
        testo_pensiero = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                testo_pensiero = block.text
            elif block.type == "tool_use":
                tool_calls.append(block)
        
        # Mostra il ragionamento di Claude
        if testo_pensiero:
            with log_container:
                st.markdown(f'<div class="thinking">💭 {testo_pensiero}</div>', 
                           unsafe_allow_html=True)
        
        # Se Claude non usa nessun tool → ha finito
        if response.stop_reason == "end_turn" or not tool_calls:
            risposta_finale = testo_pensiero
            break
        
        # Esegui ogni tool richiesto da Claude
        tool_results = []
        for tool_call in tool_calls:
            nome_tool = tool_call.name
            params = tool_call.input
            
            with log_container:
                st.markdown(
                    f'<div class="tool-call">🔧 <b>{nome_tool}</b>({json.dumps(params, ensure_ascii=False)})</div>',
                    unsafe_allow_html=True
                )
            
            # Esegui il tool
            risultato = esegui_tool(nome_tool, params)
            
            with log_container:
                st.markdown(
                    f'<div class="tool-result">✓ {risultato[:500]}{"..." if len(risultato) > 500 else ""}</div>',
                    unsafe_allow_html=True
                )
            
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": risultato
            })
        
        # Aggiorna la storia con risposta di Claude + risultati tool
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    
    return risposta_finale

# ─── UI principale ────────────────────────────────────────────
col_sinistra, col_destra = st.columns([1, 1])

with col_sinistra:
    st.markdown("### 🎯 Obiettivo")
    
    if "pending_goal" in st.session_state:
        goal_value = st.session_state.pop("pending_goal")
    else:
        goal_value = ""
    
    obiettivo = st.text_area(
        "Cosa vuoi che l'agente faccia?",
        value=goal_value,
        height=120,
        placeholder="Es: Cerca le ultime notizie sull'AI e fammi un riassunto..."
    )
    
    avvia = st.button("🚀 Avvia agente", type="primary", use_container_width=True)

with col_destra:
    st.markdown("### 🔄 Come funziona il loop agente")
    st.markdown("""
    ```
    Obiettivo
        ↓
    Claude pensa → sceglie tool
        ↓
    Tool viene eseguito
        ↓
    Risultato torna a Claude
        ↓
    Claude pensa ancora...
        ↓
    Risposta finale
    ```
    """)

st.markdown("---")

# ─── Esecuzione agente ────────────────────────────────────────
if avvia:
    if not api_key:
        st.error("Inserisci la API Key nella sidebar.")
    elif not obiettivo.strip():
        st.warning("Scrivi un obiettivo prima di avviare l'agente.")
    else:
        st.markdown("### 🤖 Esecuzione agente")
        
        col_log, col_result = st.columns([1, 1])
        
        with col_log:
            st.markdown("#### 📋 Log passi")
            log_container = st.container()
        
        with col_result:
            st.markdown("#### ✅ Risposta finale")
            result_placeholder = st.empty()
        
        with st.spinner("L'agente sta lavorando..."):
            risposta = esegui_agente(
                obiettivo, 
                api_key, 
                max_steps,
                log_container
            )
        
        with col_result:
            if risposta:
                result_placeholder.markdown(risposta)
            else:
                result_placeholder.info("L'agente ha completato i passi. Controlla il log per i dettagli.")
        
        # Salva nella sessione per mostrare la storia
        if "agent_history" not in st.session_state:
            st.session_state.agent_history = []
        st.session_state.agent_history.append({
            "obiettivo": obiettivo,
            "risposta": risposta
        })

# ─── Storico esecuzioni ───────────────────────────────────────
if "agent_history" in st.session_state and st.session_state.agent_history:
    st.markdown("---")
    st.markdown("### 📜 Storico esecuzioni")
    for i, item in enumerate(reversed(st.session_state.agent_history[-3:]), 1):
        with st.expander(f"Obiettivo: {item['obiettivo'][:60]}..."):
            st.markdown(item["risposta"])