"""
Streamlit front-end for the AI Image Forensic Detector.

Layout
------
- Hero + 3-step How it Works
- Sidebar: agentic patterns & model roster
- Two-column workspace: upload (left) | analyze / report (right)
- Live agent status + expandable RAG / analyst transparency
"""

import importlib
import inspect

import streamlit as st

# Force a fresh load of agents.py so Streamlit never keeps a stale signature
import agents as _agents_module

importlib.reload(_agents_module)
from agents import orchestrator_router

# Hard guard: fail loudly if the loaded signature drifts from what the UI expects
_expected = {"query", "image_data", "progress_callback"}
_actual = set(inspect.signature(orchestrator_router).parameters)
if not _expected.issubset(_actual):
    st.error(
        "Stale `agents` module loaded. Stop the Streamlit server (Ctrl+C) and restart "
        "with `streamlit run app.py`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ForensicSight | AI Image Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual system — forensic / lab aesthetic (slate + teal, not purple defaults)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap');

    :root {
        --ink: #0f1c24;
        --muted: #5b6b75;
        --panel: #f4f7f8;
        --line: #d7e0e5;
        --accent: #0d7377;
        --accent-deep: #095456;
        --warn: #c45c26;
    }

    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    /* Soft atmospheric background */
    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, #d9eef0 0%, transparent 55%),
            radial-gradient(900px 400px at 100% 0%, #e8eef2 0%, transparent 50%),
            linear-gradient(180deg, #f7fafb 0%, #eef3f5 100%);
    }

    .hero-brand {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: clamp(2.4rem, 4vw, 3.4rem);
        line-height: 1.05;
        margin: 0.2rem 0 0.4rem 0;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    .hero-sub {
        font-size: 1.08rem;
        color: var(--muted);
        max-width: 720px;
        margin-bottom: 1.4rem;
    }

    .step-card {
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
        backdrop-filter: blur(6px);
    }

    .step-num {
        display: inline-block;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.35rem;
    }

    .step-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }

    .step-copy {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.45;
        margin: 0;
    }

    .panel {
        background: rgba(255,255,255,0.8);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.15rem 1.25rem 1.35rem 1.25rem;
    }

    .panel-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.75rem;
    }

    .report-box {
        background: #ecf6f4;
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-top: 0.5rem;
    }

    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 0.85rem;
        padding: 1.5rem 0 0.5rem 0;
    }

    /* Tighten default Streamlit spacing a bit */
    div[data-testid="stVerticalBlock"] > div:has(> div.panel) {
        margin-top: 0.25rem;
    }

    /* Primary button accent */
    .stButton > button[kind="primary"] {
        background-color: var(--accent);
        border-color: var(--accent);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-deep);
        border-color: var(--accent-deep);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — system / assignment transparency
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### System Overview")
    st.caption("Agentic AI patterns used in this application")

    st.markdown(
        """
**Pattern 1 — Router**  
Orchestrator receives the image and routes it into the forensic pipeline.

**Pattern 2 — Tool-Use**  
Analyst queries a Chroma RAG index built from research PDFs.

**Pattern 3 — Orchestrator–Worker & Reflection**  
Analyst inspects the image; Reporter critiques findings and writes the verdict.
        """
    )

    st.divider()
    st.markdown("### Models")
    st.markdown(
        """
| Role | Model |
|------|-------|
| Routing | `llama-3.1-8b-instant` |
| Vision Analyst | `qwen/qwen3.6-27b` |
| Reporter | `llama-3.3-70b-versatile` |
        """
    )

    st.divider()
    st.info("Set a valid `GROQ_API_KEY` in your `.env` file before running analysis.")
    st.caption("IT41043 · Intelligent Systems · Agentic AI Assignment")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-brand">ForensicSight</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">'
    "Upload a photo. An agentic AI team cross-checks it against a curated research "
    "knowledge base and returns a clear forensic verdict on likely AI-generation artifacts."
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# How it Works — 3 steps
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="step-card">
          <div class="step-num">Step 01</div>
          <div class="step-title">Upload Image</div>
          <p class="step-copy">Drop a JPG, JPEG, or PNG. We compress it locally before any model call.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="step-card">
          <div class="step-num">Step 02</div>
          <div class="step-title">AI Analyzes vs Knowledge Base</div>
          <p class="step-copy">The Analyst retrieves technical guidelines via RAG, then inspects the image with a vision model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="step-card">
          <div class="step-num">Step 03</div>
          <div class="step-title">Get Forensic Report</div>
          <p class="step-copy">The Reporter reflects on the findings and delivers a plain-language final verdict.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")  # breathing room
st.divider()

# ---------------------------------------------------------------------------
# Workspace — upload | analyze
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Evidence Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="The image is resized and compressed before being sent to the vision model.",
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption=f"Preview · {uploaded_file.name}",
            use_container_width=True,
        )
    else:
        st.caption("No image selected yet. Upload evidence to unlock analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Forensic Workspace</div>', unsafe_allow_html=True)

    run_clicked = st.button(
        "Run Forensic Analysis",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if uploaded_file is None:
        st.caption("Upload an image on the left to enable the analysis button.")

    # ---- Run pipeline with live agent transparency ----
    if run_clicked:
        if uploaded_file is None:
            st.warning("Please upload a JPG, JPEG, or PNG image before running analysis.")
        else:
            image_bytes = uploaded_file.getvalue()
            step_labels = {
                "route": "🧭 **Orchestrator** — receiving image and routing to workers…",
                "rag": "🕵️ **Analyst** — checking the RAG knowledge base…",
                "vision": "🖼️ **Analyst** — calling vision model with image + context…",
                "report": "📝 **Reporter** — reflecting on findings and drafting verdict…",
                "done": "✅ **Pipeline** — agents finished handoff.",
            }

            try:
                with st.status("Agentic pipeline running…", expanded=True) as status:

                    def on_progress(step: str, detail: str = "") -> None:
                        # Stream each agent milestone into the status panel in real time
                        st.write(step_labels.get(step, detail or step))
                        if detail and step in ("rag", "vision", "report"):
                            st.caption(detail)

                    result = orchestrator_router(
                        image_data=image_bytes,
                        progress_callback=on_progress,
                    )

                    if not result.get("ok"):
                        status.update(label="Pipeline failed", state="error")
                        st.error(result.get("error") or "Unknown error.")
                    else:
                        status.update(label="Agents finished successfully", state="complete")

                if result.get("ok"):
                    # Transparency: show what RAG actually retrieved
                    with st.expander("📚 RAG context used by the Analyst", expanded=False):
                        context = result.get("context_used") or "_No context retrieved._"
                        st.markdown(context)

                    with st.expander("🔬 Raw Analyst findings", expanded=False):
                        findings = result.get("findings") or "_No findings returned._"
                        st.markdown(findings)

                    st.success("Analysis complete")
                    st.markdown("#### Final Forensic Report")
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(result.get("report", ""))
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as exc:
                st.error(f"An error occurred during execution: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="footer-note">Built for IT41043 Intelligent Systems (Agentic AI) · '
    "Groq · LangChain · Chroma RAG</p>",
    unsafe_allow_html=True,
)
