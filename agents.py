"""
Agentic workflow for AI-image forensic analysis.

Architecture
------------
1. Orchestrator (Router)     -> Groq (llama-3.1-8b-instant)
2. Forensic Analyst (Worker) -> RAG + Map-Reduce tiling via Groq vision
3. Reporter (Worker)         -> Groq (llama-3.3-70b-versatile) reflection + verdict

Vision tiling uses the official `groq` Python SDK for multimodal (image + text) payloads.
"""

import os
import base64
import io
import time
from typing import Callable, List, Optional

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
# Get VISION_MODEL from the environment, defaulting to a known valid Groq vision model.
# This prevents forcing qwen which has TPD rate limits.
VISION_MODEL = os.environ.get("VISION_MODEL", "llama-3.2-90b-vision-preview")
# Per-tile budget: high enough for forensic detail, low enough for TPM limits
TILE_MAX_SIDE = 300
TILE_JPEG_QUALITY = 75
RAG_CONTEXT_CHAR_LIMIT = 1000
COMBINED_FINDINGS_CHAR_LIMIT = 6000
QUADRANT_LABELS = ("top-left", "top-right", "bottom-left", "bottom-right")
TILE_PAUSE_SECONDS = 16  # pause between Groq vision tile calls

# ---------------------------------------------------------------------------
# Models (lightweight — safe to construct on every Streamlit reload)
# ---------------------------------------------------------------------------
router_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

reasoning_model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
)

# Official Groq client — multimodal vision tiling (uses GROQ_API_KEY)
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

def _encode_tile_jpeg(tile: Image.Image) -> str:
    """Resize a single tile if needed and return a base64 JPEG string."""
    width, height = tile.size
    longest = max(width, height)
    if longest > TILE_MAX_SIDE:
        scale = TILE_MAX_SIDE / float(longest)
        tile = tile.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    tile.save(buffer, format="JPEG", quality=TILE_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def split_image_into_tiles(image_data: bytes) -> List[str]:
    """Split an image into 4 equal quadrants and return them as base64 JPEG strings.

    Order: top-left, top-right, bottom-left, bottom-right.
    Each tile is lightly resized/compressed so a single high-res quadrant stays
    within free-tier vision payload limits while preserving useful local detail.
    """
    with Image.open(io.BytesIO(image_data)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        mid_x, mid_y = width // 2, height // 2
        boxes = [
            (0, 0, mid_x, mid_y),            # top-left
            (mid_x, 0, width, mid_y),        # top-right
            (0, mid_y, mid_x, height),       # bottom-left
            (mid_x, mid_y, width, height),   # bottom-right
        ]

        return [_encode_tile_jpeg(img.crop(box)) for box in boxes]


def _notify(callback: Optional[Callable], step: str, detail: str = "") -> None:
    """Optional UI progress hook (used by Streamlit status panels)."""
    if callback is not None:
        callback(step, detail)


def _analyze_tile(
    tile_b64: str,
    quadrant_index: int,
    context: str,
) -> str:
    """Map step: one Groq vision call for a single high-res quadrant."""
    label = QUADRANT_LABELS[quadrant_index - 1]
    prompt = f"""
You are a Forensic Micro-Artifact Inspector specializing in AI-generated imagery and digital art.

CONTEXT — crop edges (IGNORE completely):
You are analyzing ONE high-resolution digitally cropped quadrant (1/4th) of a larger image.
Straight vertical/horizontal borders are from our Python cropping script, NOT AI stitching,
collage, or compositing. Ignore them completely. Do not complain about missing parts of the scene.

AVOID FALSE POSITIVES — real smartphone selfies:
Real human faces captured with smartphone cameras often show phone flash lighting, slight JPEG
compression, noise, mild blur, beauty-filter softening, or natural skin/dental variations.
These are NORMAL. Do NOT flag ordinary smartphone characteristics as AI generation.

FLAG ONLY unmistakable AI / digital-art signals:
1. True AI anatomical flaws: melted or extra fingers, fused/warped limbs, impossible joints,
   deformed facial geometry, warped eyes/teeth/ears that are structural — not normal variation.
2. Unnatural generative skin: plastic/waxy pore-less smoothing that looks synthetic (beyond a
   mild phone beauty filter), smeared hairlines, duplicated texture patterns.
3. Distinct stylized digital art: clear illustration, anime, CGI, painting, or heavily
   AI-stylized render that is obviously not a real camera photograph.
4. Clear AI blending: subjects melting into backgrounds, garbled text/logos, morphing textures.

REPORTING RULES (balanced, mandatory):
- If a face/person looks like a normal human captured via a smartphone camera (even with flash,
  compression, mild filter, or natural imperfections), output:
  'No AI artifacts found - consistent with a real photograph'
- Only flag when you see true AI structural flaws OR distinct digital art/illustration style.
  When flagging, explicitly report the evidence.
- Do not invent objects. Stick to what is clearly visible in THIS crop.

This is the {label} quadrant ({quadrant_index} of 4).

Technical guidelines from the knowledge base (use as hints; do not invent objects):
{context}

Output a brief forensic note for this quadrant, ending with either:
- "AI ARTIFACT DETECTED: <short evidence>" OR
- "No AI artifacts found - consistent with a real photograph"
"""
    completion = groq_client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.2,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.strip(),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{tile_b64}",
                        },
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def forensic_analyst_agent(
    image_data: bytes,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Worker agent: RAG guidelines + Map-Reduce vision tiling across 4 quadrants.

    Vision calls go through Groq (`VISION_MODEL` / GROQ_API_KEY). Router/Reporter
    also remain on Groq elsewhere in the pipeline.
    """
    _notify(progress_callback, "rag", "Retrieving forensic guidelines from the knowledge base...")
    print("-> [Forensic Analyst] Checking the knowledge base...")

    docs = _get_retriever().invoke(RAG_QUERY)
    context = "\n".join(doc.page_content for doc in docs)[:RAG_CONTEXT_CHAR_LIMIT]

    _notify(progress_callback, "vision", "Splitting image into 4 high-resolution quadrants...")
    print("-> [Forensic Analyst] Splitting image into 4 tiles (Map-Reduce via Groq)...")
    _notify(
        progress_callback,
        "vision",
        f"Analyst: Using Groq vision model `{VISION_MODEL}`",
    )
    tiles = split_image_into_tiles(image_data)

    tile_findings: List[str] = []
    for i, tile_b64 in enumerate(tiles, start=1):
        detail = f"Analyst: Analyzing quadrant {i} of 4..."
        _notify(progress_callback, "vision", detail)
        print(f"-> [Forensic Analyst] {detail}")

        findings = _analyze_tile(tile_b64, i, context)
        label = QUADRANT_LABELS[i - 1]
        tile_findings.append(f"### Quadrant {i} ({label})\n{findings}")

        if i < len(tiles):
            pause_detail = (
                f"Analyst: Analyzing quadrant {i} of 4... "
                "(Pausing 16s to respect API rate limits)"
            )
            _notify(progress_callback, "vision", pause_detail)
            print(f"-> [Forensic Analyst] {pause_detail}")
            time.sleep(16)

    # Reduce: aggregate all quadrant analyses into one findings blob for the Reporter
    combined_findings = "\n\n".join(tile_findings)
    _notify(
        progress_callback,
        "vision",
        "Analyst: Aggregating findings from all 4 quadrants...",
    )
    print("-> [Forensic Analyst] Combined findings from 4 Groq vision calls.")

    return {
        "status": "analyzed",
        "findings": combined_findings,
        "context_used": context,
    }


def reporter_agent(
    analyst_message: dict,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Worker agent: reflect on analyst findings and produce a user-facing verdict."""
    _notify(progress_callback, "report", "Reflecting on findings and drafting the final verdict...")
    print("-> [Reporter] Generating the final report...")

    findings = str(analyst_message.get("findings", ""))[:COMBINED_FINDINGS_CHAR_LIMIT]

    report_prompt = f"""
You are a Fact-Checker & Reporter for an AI-image forensics system.

The Forensic Analyst inspected the image as 4 high-resolution quadrants and provided
these combined technical findings:
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

Synthesize evidence across all quadrants. If ANY quadrant reports clear AI artifacts
(e.g., "AI ARTIFACT DETECTED"), weight that heavily — do NOT default to "Likely Real Image"
just because other quadrants looked normal. Keep the whole response concise. Use Markdown
paragraphs only — no numbered lists.
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
