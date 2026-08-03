"""
Agentic workflow for AI-image forensic analysis.

Architecture
------------
1. Orchestrator (Router)     -> decides / starts the forensic pipeline
2. Forensic Analyst (Worker) -> RAG tool-use + raw Groq vision call
3. Reporter (Worker)         -> LangChain ChatGroq reflection + final verdict

Vision analysis intentionally bypasses LangChain ChatGroq and uses the official
`groq` Python SDK, which is more reliable for multimodal (image + text) payloads.
"""

import os
import base64
import io
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Cache redirects MUST happen before HuggingFace / sentence-transformers import
# ---------------------------------------------------------------------------
_HF_CACHE = os.path.abspath("./.hf_cache")
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_HF_CACHE, "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(_HF_CACHE, "sentence_transformers")

from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAG_QUERY = "technical artifacts found in AI generated images"
VISION_MODEL = "qwen/qwen3.6-27b"
IMAGE_MAX_SIDE = 256
IMAGE_JPEG_QUALITY = 60
RAG_CONTEXT_CHAR_LIMIT = 1000

# ---------------------------------------------------------------------------
# Models (lightweight — safe to construct on every Streamlit reload)
# ---------------------------------------------------------------------------
router_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

reasoning_model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
)

# Official Groq client — used EXCLUSIVELY for multimodal vision analysis
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Lazy RAG handles (heavy SentenceTransformer load deferred until first use)
_retriever = None


def _get_retriever():
    """Lazy-init Chroma retriever so Streamlit can hot-reload agents.py cleanly."""
    global _retriever
    if _retriever is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )
        # k=1 keeps retrieved text small enough for the vision context window
        _retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    return _retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_image(image_data: bytes) -> str:
    """Resize / compress an uploaded image and return a base64 JPEG string."""
    with Image.open(io.BytesIO(image_data)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        longest = max(width, height)
        if longest > IMAGE_MAX_SIDE:
            scale = IMAGE_MAX_SIDE / float(longest)
            img = img.resize(
                (int(width * scale), int(height * scale)),
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _notify(callback: Optional[Callable], step: str, detail: str = "") -> None:
    """Optional UI progress hook (used by Streamlit status panels)."""
    if callback is not None:
        callback(step, detail)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def forensic_analyst_agent(
    image_data: bytes,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Worker agent: retrieve RAG guidelines, then analyze the image via Groq vision.

    Parameters
    ----------
    image_data:
        Raw image bytes from the uploader (passed through from orchestrator_router).
    progress_callback:
        Optional UI status callback.
    """
    _notify(progress_callback, "rag", "Retrieving forensic guidelines from the knowledge base...")
    print("-> [Forensic Analyst] Checking the knowledge base...")

    docs = _get_retriever().invoke(RAG_QUERY)
    context = "\n".join(doc.page_content for doc in docs)[:RAG_CONTEXT_CHAR_LIMIT]

    _notify(progress_callback, "vision", "Sending compressed image + RAG context to the vision model...")
    print("-> [Forensic Analyst] Calling raw Groq vision API...")

    # Ensure the vision model actually receives the image payload
    base64_image = _preprocess_image(image_data)

    completion = groq_client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a Forensic Image Analyst specializing in AI-generated imagery.\n\n"
                            f"Technical guidelines from the knowledge base:\n{context}\n\n"
                            "IMPORTANT: The image you are analyzing has been highly compressed to 256px "
                            "and may be a photograph of a digital screen. DO NOT confuse standard JPEG "
                            "compression artifacts, pixelation, or screen moiré patterns (grids/lines) "
                            "with AI-generation artifacts. Be highly skeptical before claiming an image "
                            "is AI-generated based on blurriness alone.\n\n"
                            "Analyze the attached image for genuine AI-generation artifacts "
                            "(unnatural textures, lighting, anatomy, text, reflections, etc.). "
                            "Provide a brief structured technical analysis and an overall "
                            "likelihood that the image is AI-generated."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )

    findings = completion.choices[0].message.content or ""

    return {
        "status": "analyzed",
        "findings": findings,
        "context_used": context,
    }


def reporter_agent(
    analyst_message: dict,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Worker agent: reflect on analyst findings and produce a user-facing verdict."""
    _notify(progress_callback, "report", "Reflecting on findings and drafting the final verdict...")
    print("-> [Reporter] Generating the final report...")

    findings = str(analyst_message.get("findings", ""))[:2000]

    report_prompt = f"""
You are a Fact-Checker & Reporter for an AI-image forensics system.

The Forensic Analyst provided these technical findings:
{findings}

Write the final report using the Bottom Line Up Front (BLUF) principle.
Do not add any preamble before the first line.

Task:
- First line: Output ONLY the final verdict in bold (e.g., **Verdict: Likely AI-Generated** or **Verdict: Likely Real Image**). Do not use numbers or repeat the word 'Verdict'.
- Second paragraph: Provide the 'Critique of Findings'.
- Third paragraph: Provide a simple 'Detailed Explanation'.

Allowed verdicts (choose exactly one):
- **Verdict: Likely AI-Generated**
- **Verdict: Likely Real Image**
- **Verdict: Inconclusive/Uncertain**

Keep the whole response concise. Use Markdown paragraphs only — no numbered lists.
"""

    response = reasoning_model.invoke(report_prompt)
    return response.content


def orchestrator_router(query=None, image_data=None, progress_callback=None) -> dict:
    """Router / Orchestrator: validate input and run Analyst -> Reporter pipeline.

    Signature is intentionally synchronized with app.py:
        orchestrator_router(query=None, image_data=None, progress_callback=None)
    """
    # Prefer explicit image_data from Streamlit; allow legacy bytes via query
    resolved_image = image_data
    if resolved_image is None and isinstance(query, (bytes, bytearray)):
        resolved_image = bytes(query)

    print("\n[Orchestrator] Received image for forensic analysis")
    _notify(progress_callback, "route", "Orchestrator routing image to the forensic analyst...")

    if not resolved_image:
        return {
            "ok": False,
            "error": "Please upload a valid image (JPG, JPEG, or PNG).",
            "report": "",
            "findings": "",
            "context_used": "",
        }

    # Pass image_data through to the analyst so Groq vision receives the payload
    analyst_msg = forensic_analyst_agent(
        image_data=resolved_image,
        progress_callback=progress_callback,
    )
    final_report = reporter_agent(
        analyst_msg,
        progress_callback=progress_callback,
    )

    _notify(progress_callback, "done", "Pipeline complete.")

    return {
        "ok": True,
        "error": "",
        "report": final_report,
        "findings": analyst_msg.get("findings", ""),
        "context_used": analyst_msg.get("context_used", ""),
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agents.py <path-to-image>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as handle:
        result = orchestrator_router(image_data=handle.read())

    print("\n" + "=" * 50)
    if result["ok"]:
        print(result["report"])
    else:
        print(result["error"])
    print("=" * 50 + "\n")
