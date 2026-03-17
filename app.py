import streamlit as st
import chromadb
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY     = os.getenv("OPENAI_API_KEY")
DB_PATH     = str(Path(__file__).parent / "chroma_db")
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL   = "gpt-4o"
TOP_K       = 8

client = OpenAI(api_key=API_KEY)

@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return chroma_client.get_collection("family_offices")

def embed_query(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding

def retrieve(query, top_k=TOP_K):
    collection = load_collection()
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results["documents"][0], results["metadatas"][0], results["distances"][0]

def generate_answer(query, documents):
    context = "\n\n---\n\n".join(documents)
    system_prompt = """You are a family office intelligence analyst.
Answer questions strictly based on the family office dataset provided as context.
Be specific, cite family office names, and structure answers clearly.
If context does not contain enough information, say so honestly.
Never fabricate details not present in the context."""
    user_prompt = f"""Based on the following family office records, answer this query:

QUERY: {query}

CONTEXT:
{context}

Provide a clear, structured answer with specific names, AUM figures, contact details, and relevant intelligence."""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolarityIQ — Family Office Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {
        background-color: #0f1117 !important;
        font-family: 'Inter', sans-serif;
        color: #e2e8f0 !important;
    }

    /* ── HEADER ── */
    .header-wrap {
        background: linear-gradient(135deg, #1A1A2E 0%, #162447 100%);
        border: 1px solid rgba(37, 99, 235, 0.3);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.15);
    }
    .header-wrap h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-wrap p {
        color: #94a3b8 !important;
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
        font-weight: 300;
    }
    .header-badge {
        display: inline-block;
        background: rgba(37, 99, 235, 0.2);
        border: 1px solid rgba(37, 99, 235, 0.4);
        color: #60a5fa !important;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── STAT BOXES ── */
    .stat-box {
        background: #1e2235;
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .stat-box:hover {
        border-color: rgba(37, 99, 235, 0.5);
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.1);
        transform: translateY(-2px);
    }
    .stat-num {
        font-size: 2rem;
        font-weight: 700;
        color: #2563EB !important;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .stat-label {
        color: #64748b !important;
        font-size: 0.82rem;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* ── SECTION HEADERS ── */
    .section-header {
        color: #e2e8f0 !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* ── INPUT & SELECT ── */
    .stTextInput > div > div > input {
        background: #1e2235 !important;
        border: 1.5px solid rgba(37, 99, 235, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #475569 !important;
    }
    .stTextInput label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div {
        background: #1e2235 !important;
        border: 1.5px solid rgba(37, 99, 235, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    .stSelectbox label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox svg { fill: #64748b !important; }

    /* ── BUTTON ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1d4ed8 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 2rem !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ── SLIDER ── */
    .stSlider label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* ── ANSWER BOX — uses st.markdown so styling goes on wrapper ── */
    .answer-container {
        background: #1e2235;
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-left: 4px solid #2563EB;
        border-radius: 14px;
        padding: 0.25rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(37, 99, 235, 0.08);
    }
    .answer-container p,
    .answer-container li,
    .answer-container h1,
    .answer-container h2,
    .answer-container h3,
    .answer-container strong {
        color: #e2e8f0 !important;
    }
    .answer-container ul, .answer-container ol {
        color: #e2e8f0 !important;
        padding-left: 1.5rem;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        background: #1e2235 !important;
        border: 1px solid rgba(37, 99, 235, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        padding: 0.9rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(37, 99, 235, 0.4) !important;
        background: #252840 !important;
    }
    .streamlit-expanderHeader p { color: #e2e8f0 !important; }
    .streamlit-expanderContent {
        background: #1a1d2e !important;
        border: 1px solid rgba(37, 99, 235, 0.15) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.25rem 1.5rem !important;
        color: #cbd5e1 !important;
    }
    .streamlit-expanderContent p,
    .streamlit-expanderContent div,
    .streamlit-expanderContent span,
    .streamlit-expanderContent strong {
        color: #cbd5e1 !important;
    }

    /* ── ALERTS ── */
    .stAlert {
        background: #1e2235 !important;
        border: 1px solid rgba(37, 99, 235, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    .stAlert p { color: #e2e8f0 !important; }

    /* ── CAPTION ── */
    .stCaption, .stCaption p {
        color: #475569 !important;
        font-size: 0.85rem !important;
    }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: rgba(37, 99, 235, 0.15) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── MARKDOWN TEXT ── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] strong {
        color: #e2e8f0 !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2563EB; border-radius: 10px; }

    /* ── FOOTER ── */
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #e6eaed !important;
        font-size: 0.82rem;
        font-weight: 300;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <h1>🏦 PolarityIQ</h1>
    <p>Family Office Intelligence — Natural Language Query Interface</p>
    <div class="header-badge">266 Records · 31 Fields · RAG + GPT-4o · Real-World Data</div>
</div>
""", unsafe_allow_html=True)

# ── STATS ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, num, label in zip(
    [c1, c2, c3, c4, c5],
    ["266", "31", "12+", "$225B", "GPT-4o"],
    ["Family Offices", "Intel Fields", "Countries", "Largest AUM", "AI Engine"]
):
    with col:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# ── QUERY INTERFACE ───────────────────────────────────────────────────────────
st.markdown('<p class="section-header">🔍 Query the Intelligence Database</p>', unsafe_allow_html=True)

examples = [
    "Which family offices focus on AI with check sizes above $10M?",
    "Show me all SFOs in the Middle East that made direct investments recently",
    "Which European family offices co-invest frequently in clean energy?",
    "Find family offices in Texas focused on energy with contact emails",
    "Which family offices have AUM above $50B and invest in technology?",
    "Show me impact-focused family offices with high co-invest frequency",
    "Which family offices were founded by tech entrepreneurs?",
    "List all family offices in Singapore or Hong Kong",
    "Which family offices have verified contact emails?",
]

selected = st.selectbox(
    "Try an example query:",
    ["— select an example or type your own below —"] + examples,
    key="example_select"
)

if selected != "— select an example or type your own below —":
    query = selected
    st.info(f"**Running:** {query}")
else:
    query = st.text_input(
        "Or type your own query:",
        placeholder="e.g. Which family offices focus on AI with check sizes above $10M?",
        key="query_input"
    )

col_btn, col_k = st.columns([4, 1])
with col_btn:
    search_btn = st.button("🔍  Search Intelligence Database", type="primary", use_container_width=True)
with col_k:
    top_k = st.slider("Results", 3, 15, 8)

# ── RESULTS ───────────────────────────────────────────────────────────────────
if search_btn and query and query.strip():
    with st.spinner("Querying 266 family offices..."):
        try:
            documents, metadatas, distances = retrieve(query, top_k=top_k)
            answer = generate_answer(query, documents)

            # ── INTELLIGENCE REPORT — rendered as native markdown ──
            st.markdown('<p class="section-header">💡 Intelligence Report</p>', unsafe_allow_html=True)
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── SOURCE RECORDS ──
            st.markdown(f'<p class="section-header">📋 Source Records ({len(documents)} retrieved)</p>', unsafe_allow_html=True)
            st.caption("Records retrieved from the vector database — ranked by semantic relevance to your query")

            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                relevance = round((1 - dist) * 100, 1)
                name  = meta.get('name', 'Unknown')
                ftype = meta.get('type', '')
                aum   = meta.get('aum', '')

                with st.expander(f"#{i+1}  {name}   ·   {ftype}   ·   {aum}   ·   {relevance}% match"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**🌍 Country:** {meta.get('country','—')}")
                        st.markdown(f"**📊 Sector:** {meta.get('sector','—')}")
                        st.markdown(f"**💰 Check Size:** {meta.get('check','—')}")
                        st.markdown(f"**🎯 Style:** {meta.get('style','—')}")
                    with col_b:
                        st.markdown(f"**🤝 Co-Invest:** {meta.get('coinvest','—')}")
                        st.markdown(f"**✅ Direct Active:** {meta.get('direct','—')}")
                        email = meta.get('email', '')
                        if email:
                            st.markdown(f"**📧 Email:** `{email}`")
                    st.markdown("---")
                    st.code(doc[:1000] + "..." if len(doc) > 1000 else doc, language=None)

        except Exception as e:
            st.error(f"Error: {str(e)}")

elif search_btn:
    st.warning("Please select an example or type a query first.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    OpenAI text-embedding-3-small · GPT-4o · ChromaDB · Streamlit<br>
    PolarityIQ Differentiator — Task 2 · Hadia Tahir
</div>
""", unsafe_allow_html=True)