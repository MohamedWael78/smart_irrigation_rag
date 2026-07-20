import os

import streamlit as st
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from retriever import setup_advanced_retriever
from tools import (
    calculate_drip_irrigation,
    get_reference_evapotranspiration,
    lookup_crop_coefficient,
)
from memory import build_chat_history
from streaming import stream_agent_response
from citations import parse_citations, render_citation_cards_html

# --- 1. Load API key ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Optional: LangSmith tracing for observability. Purely opt-in via secrets.
if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "aquamind-ai")

# --- 2. Page config + design tokens ---
st.set_page_config(page_title="AquaMind AI", page_icon="\U0001f4a7", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        /* Canvas: pale mist over a faint blueprint grid */
        --c-canvas-a:#f4f7f5; --c-canvas-b:#eef4f2; --c-canvas-c:#eaf6f5;
        --c-ink:#0e1b1a; --c-muted:#5b6b67; --c-border:#dbe6e2;
        /* Accents: water (flow) + crop (growth); clay reserved for alerts only */
        --c-flow:#0ea5b7; --c-flow-dark:#0b8494;
        --c-growth:#67a63c; --c-growth-dark:#4f7f2c;
        --c-clay:#c97a3d;
        /* Console (sidebar): deep pine-black instrument panel */
        --c-console-a:#071312; --c-console-b:#0b211f; --c-console-c:#0a2e2b;
        --c-console-line: rgba(14,165,183,0.28);
        --font-display:'Space Grotesk', sans-serif;
        --font-body:'Plus Jakarta Sans', sans-serif;
        --font-mono:'JetBrains Mono', monospace;
    }
    * { font-family: var(--font-body); }
    h1, h2, h3, h4 { font-family: var(--font-display) !important; }
    code, .stCodeBlock, [data-testid="stMetricValue"] { font-family: var(--font-mono) !important; }

    /* Faint blueprint grid under the usual color wash -- grounded in the
       subject: irrigation design literally happens on schematics. */
    .stApp {
        background:
            linear-gradient(rgba(14,165,183,0.05) 1px, transparent 1px) 0 0 / 100% 34px,
            linear-gradient(90deg, rgba(14,165,183,0.05) 1px, transparent 1px) 0 0 / 34px 100%,
            radial-gradient(circle at top right, var(--c-canvas-c) 0%, var(--c-canvas-b) 40%, var(--c-canvas-a) 100%);
    }

    .brand-row { display: flex; align-items: center; justify-content: center; gap: 14px; margin-bottom: 2px; }
    .brand-mark { flex: 0 0 auto; filter: drop-shadow(0 3px 8px rgba(14,165,183,0.35)); }
    .main-title {
        font-family: var(--font-display) !important;
        color: var(--c-ink);
        font-weight: 700; font-size: 2.6rem !important; letter-spacing: -0.5px; margin: 0;
    }
    .main-title .accent { color: var(--c-flow); }

    .subtitle { text-align: center; color: var(--c-muted); font-size: 1.05rem; font-weight: 500; margin-top: 2px; }

    /* Signature element: an animated flow-meter -- a droplet travelling a
       graduated pipe -- standing in for live hydraulic instrumentation. */
    .flow-meter { display: flex; justify-content: center; margin: 14px auto 6px auto; }
    .flow-meter svg { overflow: visible; }
    .flow-dot { animation: none; }
    @media (prefers-reduced-motion: reduce) {
        .flow-meter animateMotion { display: none; }
    }

    /* Status strip: console-style readout of what's currently active */
    .status-strip {
        display: flex; justify-content: center; flex-wrap: wrap; gap: 18px;
        margin: 4px 0 22px 0; padding: 8px 0;
        border-top: 1px solid var(--c-border); border-bottom: 1px solid var(--c-border);
    }
    .status-item {
        display: flex; align-items: center; gap: 6px;
        font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.06em;
        color: var(--c-muted); text-transform: uppercase;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; box-shadow: 0 0 0 3px rgba(14,165,183,0.12); }

    [data-testid="stChatMessage"] {
        position: relative;
        background-color: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(14, 27, 26, 0.04);
        padding: 22px 20px 18px 20px; margin-bottom: 20px; margin-top: 8px;
        border: 1px solid var(--c-border);
        transition: transform 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(14, 27, 26, 0.07); }
    [data-testid="stChatMessage"][data-test-user="assistant"] { border-left: 3px solid var(--c-growth); }
    [data-testid="stChatMessage"][data-test-user="user"] { border-left: 3px solid var(--c-flow); background-color: rgba(248, 250, 252, 0.92); }
    [data-testid="stChatMessage"][data-test-user="assistant"]::before {
        content: "AGENT"; position: absolute; top: -9px; left: 16px;
        font-family: var(--font-mono); font-size: 0.6rem; font-weight: 600; letter-spacing: 0.14em;
        background: var(--c-growth); color: white; padding: 2px 8px; border-radius: 5px;
    }
    /* --- NUMBER INPUTS (Lat/Lon) --- */
    /* Force the inner container background to be dark */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] > div > div {
        background-color: rgba(7, 19, 18, 0.9) !important;
        border: 1px solid rgba(14, 165, 183, 0.4) !important;
        border-radius: 8px !important;
    }
    /* Force the typed text to be bright cyan so it's visible on the dark background */
    [data-testid="stSidebar"] input[type="number"] {
        color: #7be0ea !important;
        -webkit-text-fill-color: #7be0ea !important; /* Stops browser from overriding */
        background-color: transparent !important;
        font-family: var(--font-mono) !important;
        font-weight: 600 !important;
    }
    /* Force the up/down buttons to be visible */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        color: #7be0ea !important;
        background-color: rgba(14, 165, 183, 0.1) !important;
    }

    /* --- FILE UPLOADER --- */
    /* Force the dropzone box to be dark */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(7, 19, 18, 0.9) !important;
        border: 1.5px dashed rgba(14, 165, 183, 0.5) !important;
        border-radius: 12px !important;
    }
    /* Force text inside uploader to be light/visible */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p {
        color: #b8d8d4 !important;
        -webkit-text-fill-color: #b8d8d4 !important;
    }
    /* Force the "Browse files" button to be visible */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: rgba(14, 165, 183, 0.2) !important;
        color: #0ea5b7 !important;
        border: 1px solid #0ea5b7 !important;
    }

    .stButton > button {
        background-color: #ffffff; color: var(--c-ink); border: 1px solid var(--c-border);
        border-radius: 10px; padding: 10px 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.2s ease; text-align: left; font-weight: 500; font-size: 0.92rem;
    }
    .stButton > button:hover { border-color: var(--c-flow); background-color: #ecfbfa; color: var(--c-flow-dark); box-shadow: 0 4px 12px rgba(14,165,183,0.14); }

    .streamlit-expanderHeader { background-color: #f1f6f5 !important; border-radius: 8px !important; border: 1px solid var(--c-border) !important; color: var(--c-ink) !important; }

    [data-testid="stChatInput"] { border-radius: 12px !important; border: 1px solid #cfe0dc !important; background-color: white !important; box-shadow: 0 10px 25px rgba(14, 27, 26, 0.07) !important; }
    [data-testid="stChatInput"] textarea { color: var(--c-ink) !important; }

    .tool-pill {
        display: inline-block; font-family: var(--font-mono); font-size: 0.7rem;
        background: rgba(14,165,183,0.08); color: var(--c-flow); border: 1px solid rgba(14,165,183,0.25);
        border-radius: 6px; padding: 2px 9px; margin: 2px 5px 2px 0;
    }

    .qa-label {
        font-family: var(--font-mono) !important; font-size: 0.66rem !important;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #7be0ea !important; margin: 16px 0 6px 2px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header: wordmark + custom SVG signature (flow-meter) ---
_TICKS = "".join(f'<line x1="{20 + i*26}" y1="16" x2="{20 + i*26}" y2="24" stroke="#c3d6d1" stroke-width="1.5"/>' for i in range(12))

st.markdown(f"""
<div class="brand-row">
  <svg class="brand-mark" width="40" height="40" viewBox="0 0 48 48">
    <defs>
      <linearGradient id="dropGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#0ea5b7"/><stop offset="100%" stop-color="#67a63c"/>
      </linearGradient>
    </defs>
    <path d="M24 4 C24 4 8 25 8 34 A16 16 0 0 0 40 34 C40 25 24 4 24 4 Z" fill="url(#dropGrad)"/>
    <path d="M15 31 Q24 21 33 31" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.85"/>
  </svg>
  <h1 class="main-title">AquaMind<span class="accent"> AI</span></h1>
</div>
<p class="subtitle">Next-Gen Smart Irrigation Agent &amp; Hydraulic Calculator</p>
<div class="flow-meter">
  <svg width="320" height="34" viewBox="0 0 320 34" xmlns="http://www.w3.org/2000/svg">
    <line x1="12" y1="20" x2="308" y2="20" stroke="#c3d6d1" stroke-width="2"/>
    {_TICKS}
    <circle r="5" fill="url(#dropGrad2)">
      <animateMotion dur="3.4s" repeatCount="indefinite" path="M12,20 L308,20"/>
    </circle>
    <defs>
      <linearGradient id="dropGrad2" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#0ea5b7"/><stop offset="100%" stop-color="#67a63c"/>
      </linearGradient>
    </defs>
  </svg>
</div>
<div class="status-strip">
  <span class="status-item"><span class="status-dot" style="background:#0ea5b7"></span>Hybrid Retrieval</span>
  <span class="status-item"><span class="status-dot" style="background:#67a63c"></span>Llama 3.1 &middot; Streaming</span>
  <span class="status-item"><span class="status-dot" style="background:#0ea5b7"></span>Memory On</span>
  <span class="status-item"><span class="status-dot" style="background:#c97a3d"></span>4 Tools Armed</span>
</div>
""", unsafe_allow_html=True)

# --- 3. Retriever + knowledge-base tool (custom, to preserve page metadata) ---
advanced_retriever = setup_advanced_retriever()


@tool
def search_knowledge_base(query: str):
    """Searches and returns information from FAO irrigation manuals, crop water
    requirements, soil types, and troubleshooting guides. Use this for any
    theoretical or domain-specific question. Always cite results using the
    [n] markers included in the returned context."""
    docs = advanced_retriever.invoke(query)
    if not docs:
        return "No relevant documents found in the knowledge base."
    parts = []
    for i, d in enumerate(docs, start=1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] ({src}, p.{page}): {d.page_content}")
    return "\n\n".join(parts)


tools = [
    search_knowledge_base,
    calculate_drip_irrigation,
    get_reference_evapotranspiration,
    lookup_crop_coefficient,
]

# --- 4. Sidebar (Professional Engineering Dashboard & Tool Panels) ---
with st.sidebar:
    st.markdown("### 🌿 AquaMind Console")
    st.caption("SYSTEM STATUS: ONLINE")

    # --- Agent Dashboard & Capabilities ---
    st.markdown("#### ⚙️ Agent Dashboard")
    cap_col, tech_col = st.columns(2)

    with cap_col:
        st.markdown("""
        **Capabilities**
        🧠 Hydraulic Design
        🌱 Crop Coefficients
        🛠️ Hardware Troubleshoot
        🌤️ Live Weather (ET0)
        """)
        # Dynamic document count (safely wrapped)
        try:
            # Check if vector_db is accessible to get doc count
            doc_count = len(vector_db.get()["ids"])
            st.markdown(f"📚 **Docs Loaded:** {doc_count}")
        except:
            st.markdown("📚 **Docs Loaded:** ✅")

    with tech_col:
        st.markdown("""
        **Tech Stack**  
        <span style='font-size:0.7em; background-color:rgba(14,165,183,0.15); color:#0ea5b7; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>Llama 3.1</span>
        <span style='font-size:0.7em; background-color:rgba(14,165,183,0.15); color:#0ea5b7; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>Hybrid RAG</span>
        <span style='font-size:0.7em; background-color:rgba(14,165,183,0.15); color:#0ea5b7; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>FlashRank</span>
        <span style='font-size:0.7em; background-color:rgba(14,165,183,0.15); color:#0ea5b7; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>ChromaDB</span>
        <span style='font-size:0.7em; background-color:rgba(14,165,183,0.15); color:#0ea5b7; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>4 Tools</span>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Structured Tool Panels ---
    st.markdown("#### 🧮 Engineering Tools")
    st.caption("Input parameters directly for precise calculations")

    # Drip Calculator Panel
    with st.expander("💧 Drip Calculator", expanded=False):
        drip_emitters = st.number_input("Number of Emitters", min_value=1, value=500, key="drip_emit")
        drip_flow = st.number_input("Flow Rate (L/h per emitter)", min_value=0.1, value=2.0, format="%.1f", key="drip_flow")
        drip_hours = st.number_input("Operation Hours", min_value=0.1, value=1.5, format="%.1f", key="drip_hours")
        if st.button("💧 Calculate Drip Volume", use_container_width=True):
            prompt = f"Calculate the total water volume and flow rate for a drip irrigation system with {drip_emitters} emitters, flowing at {drip_flow} L/h per emitter, operating for {drip_hours} hours."
            st.session_state.active_question = prompt

    # Soil Estimator Panel
    with st.expander("🧪 Soil Estimator", expanded=False):
        soil_types = ["Sandy", "Loamy Sand", "Sandy Loam", "Loam", "Silt Loam", "Clay Loam", "Clay"]
        selected_soil = st.selectbox("Select Soil Type", soil_types, key="soil_select")
        if st.button("🧪 Get Soil Properties", use_container_width=True):
            prompt = f"Estimate the soil water properties (field capacity, wilting point, infiltration rate) for {selected_soil} soil."
            st.session_state.active_question = prompt

    st.markdown("---")

    # --- Field Location & Vision (For ET0/Weather & Images) ---
    st.markdown("#### 🌤️ Field Context")
    st.caption("Location & images for weather/valve AI")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        field_lat = st.number_input("Latitude", value=30.0444, format="%.4f", key="lat_input")
    with col_lon:
        field_lon = st.number_input("Longitude", value=31.2357, format="%.4f", key="lon_input")
        
    if st.button("🌤️ Get ET0 Forecast", use_container_width=True):
        prompt = f"What is the ET0 forecast for my field this week based on latitude {field_lat} and longitude {field_lon}?"
        st.session_state.active_question = prompt

    st.markdown("") # Spacer
    uploaded_image = st.file_uploader("📷 Upload Field Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.markdown("---")

    # --- Knowledge Base Q&A ---
    with st.expander("💡 Knowledge Base Q&A"):
        kb_questions = [
            "Recommended emitter spacing for sandy soil?",
            "Soil moisture reads 100% but plants are wilting, why?",
            "What is the Kc value for tomatoes at mid-season?",
            "How to troubleshoot a Hunter PGV valve that won't open?",
        ]
        for q in kb_questions:
            if st.button(f"› {q}", key=q):
                st.session_state.active_question = q

    st.markdown("---")
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. LLM + Agent (memory-aware) ---
# llama-3.3-70b-versatile is being retired by Groq on 08/16/26; openai/gpt-oss-120b
# is Groq's own recommended replacement and has more reliable tool-calling.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, streaming=True)

system_prompt = (
    "You are an expert AI agent for Smart Irrigation Systems named 'AquaMind'. "
    "You have access to a knowledge base, a hydraulic calculator, a live ET0/weather "
    "tool, and a crop coefficient (Kc) lookup table. Use them to answer accurately. "
    f"Unless the user gives a different location, assume the field is at "
    f"latitude {field_lat}, longitude {field_lon} for any weather/ET0 questions. "
    "Always cite the [n] source markers from the knowledge base when you use it. "
    "Format answers with markdown, bullet points, and bold text.\n\n"
    "CRITICAL RULE: Once you execute a tool (like calculate_drip_irrigation) and receive the calculation results, "
    "DO NOT call any more tools or repeat calculations. Immediately formulate your final answer using the exact results provided by the tool."
    "HYDRAULIC & CALCULATION RULES:\n"
    "- Whenever you use a calculation tool (like calculate_drip_irrigation), you MUST explicitly display the exact raw numbers, calculated flow rates, total volumes, and metrics returned by the tool in your final response.\n"
    "- NEVER just say 'I have calculated it'. You must present the breakdown of the results clearly (e.g., Total Flow Rate, Total Volume, Runtime) using markdown tables or bold bullet points.\n\n"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15, handle_parsing_errors=True, early_stopping_method="generate")

# --- 6. Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Welcome to AquaMind! 👋 I'm your Smart Irrigation AI. I now remember our "
            "conversation and can pull real ET0/weather data, FAO crop coefficients, "
            "and calculate hydraulic parameters. How can I help?"
        ),
        "citations": [],
    }]

if "active_question" not in st.session_state:
    st.session_state.active_question = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("citations"):
            try:
                components.html(render_citation_cards_html(message["citations"]), height=140, scrolling=True)
            except:
                pass # Failsafe just in case

# --- 7. Chat Input & Logic ---
# IMPORTANT: We must define the chat input FIRST so Python knows what 'input_text' is!
input_text = st.chat_input("Ask about irrigation design, crop water requirements...")

# Handle sidebar quick actions (override input_text if a button was clicked)
if st.session_state.active_question:
    input_text = st.session_state.active_question
    st.session_state.active_question = None 

# Process the input if the user sent a message or clicked a button
if input_text:
    # 1. Display user message
    st.chat_message("user").markdown(input_text)
    
    # Check if the user uploaded an image in the sidebar
    if uploaded_image is not None:
        import base64
        from langchain_core.messages import HumanMessage
        
        # Read and encode the image
        image_bytes = uploaded_image.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Display the uploaded image in the chat
        st.chat_message("user").image(image_bytes)
        
        # Save a text placeholder to history (saving the actual image in history can crash Streamlit)
        st.session_state.messages.append({
            "role": "user", 
            "content": f"{input_text} [User uploaded an image: {uploaded_image.name}]", 
            "citations": []
        })
    else:
        # 2. Save user message to history (No image)
        st.session_state.messages.append({"role": "user", "content": input_text, "citations": []})

    # 3. Build chat history for the LLM memory
    chat_history = build_chat_history(st.session_state.messages[:-1])

    # 4. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("AquaMind Engineering Manager is working..."):
            try:
                # ==========================================
                # VISION LOGIC (If Image is Uploaded)
                # ==========================================
                if uploaded_image is not None:
                    # Format the message for the Vision LLM
                    user_content = [
                        {"type": "text", "text": input_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                    
                    # llava-v1.5-7b-4096-preview was shut down by Groq on 10/28/24;
                    # qwen/qwen3.6-27b is Groq's current (and only) supported vision model.
                    vision_llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)
                    vision_response = vision_llm.invoke([HumanMessage(content=user_content)])
                    
                    answer = vision_response.content
                    all_citations = [] # Vision models don't use the retriever tools
                
                # ==========================================
                # STANDARD AGENT LOGIC (No Image)
                # ==========================================
                else:
                    # Use standard invoke to prevent infinite looping
                    response = agent_executor.invoke(
                        {"input": input_text, "chat_history": chat_history},
                        config={"return_intermediate_steps": True}
                    )
                    
                    answer = response["output"]
                    all_citations = []
                    seen_sources = set()
                    
                    # Extract citations and tool usage from intermediate steps
                    intermediate_steps = response.get("intermediate_steps", [])
                    for step in intermediate_steps:
                        action = step[0]
                        tool_name = action.tool
                        tool_input = action.tool_input
                        
                        if tool_name == "search_knowledge_base":
                            out = step[1]
                            text = getattr(out, "content", None) or str(out)
                            for c in parse_citations(text):
                                key = (c["source"], c["page"])
                                if key not in seen_sources:
                                    seen_sources.add(key)
                                    all_citations.append(c)

                # ==========================================
                # DISPLAY AND SAVE RESPONSE (For both paths)
                # ==========================================
                # Display the final answer
                st.markdown(answer, unsafe_allow_html=True)
                
                # Display citation cards if any
                if all_citations:
                    try:
                        components.html(render_citation_cards_html(all_citations), height=140, scrolling=True)
                    except:
                        pass

                # Save assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": all_citations,
                })

            except Exception as e:
                err_text = str(e)
                if "failed_generation" in err_text or "Failed to call a function" in err_text:
                    st.error(
                        "The model had trouble formatting a tool call for that request. "
                        "Try rephrasing it more explicitly (e.g. spell out exact numbers/units "
                        "for calculator questions), or ask one thing at a time."
                    )
                else:
                    st.error(f"An error occurred: {e}")