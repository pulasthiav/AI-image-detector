# ForensicSight: Agentic AI Image Forensics & Deepfake Detection System

## 1. Project Title & Overview

**Project Name:** ForensicSight: Agentic AI Image Forensics & Deepfake Detection System.

**Option A: Real-world problem:** This project provides a practical solution assisting digital forensics teams, journalists, and security researchers in rapidly analyzing and identifying AI-generated media, structural digital manipulations, and deepfakes.

**Description:** An advanced agentic AI application designed to detect AI-generated images, digital manipulations, and deepfakes using a multi-agent pipeline, Chroma RAG knowledge base, and high-resolution map-reduce tiling. This project was developed to align strictly with the Horizon Campus IT41043 Agentic AI assignment requirements.

## 2. Architecture & Design Patterns (Section 4a)

ForensicSight leverages a robust multi-agent architecture implementing 3 distinct agentic patterns. The orchestrator logic resides in `app.py`, while the worker routing and tiling logic is handled primarily in `agents.py`.

1. **Pattern 1: Router** (`app.py`, `agents.py`)
   - The Orchestrator agent receives the image uploaded by the user and routes it efficiently into the agentic forensic pipeline.
2. **Pattern 2: Tool-Use + Map-Reduce Tiling** (`agents.py`)
   - The Analyst agent queries a Chroma RAG index for domain-specific forensic guidelines.
   - The image is split into 4 high-resolution quadrants. The Analyst inspects each quadrant utilizing separate vision model calls (Map) and then aggregates the granular findings (Reduce).
3. **Pattern 3: Orchestrator-Worker & Reflection** (`agents.py`)
   - The Analyst acts as the primary worker, returning high-resolution tile inspection findings.
   - The Reporter agent reflects upon and critiques the combined findings collectively, finalizing a comprehensive verdict.

## 3. Agent-to-Agent Communication (Section 4b)

The agents exchange structured messages using LangChain and a custom protocol to efficiently pass state (image bytes, strings, agent metadata) across the forensic pipeline.

```text
+------+      +--------------------+      +-------------------------+      +------------------+      +-----------+
| User | ---> | Router Agent       | ---> | RAG Tool & Vision       | ---> | Reporter Agent   | ---> | Final UI  |
|      |      | (Orchestration)    |      | Analyst (Map-Reduce)    |      | (Reflection)     |      | Verdict   |
+------+      +--------------------+      +-------------------------+      +------------------+      +-----------+
```

## 4. Model Selection Strategy & Comparison Table (Section 4c)

Our model strategy deliberately relies on a multi-model setup to optimize latency, cost, and multimodal reasoning capabilities.

| Sub-task                        | Model (Provider)                      | Why Chosen (Latency, Cost, Context, Reasoning)                                                                                 |
| ------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Intent Routing                  | `llama-3.1-8b-instant` (Groq)         | Ultra-low latency, near-zero cost, highly efficient for fast intent classification and routing decisions.                      |
| High-Res Vision Tiling Analysis | Groq Vision Model (4-tile Map-Reduce) | High-speed multimodal processing across image quadrants to detect structural visual artifacts.                                 |
| Reporting & Synthesis           | `llama-3.3-70b-versatile` (Groq)      | Superior reasoning capacity and large context window required for deep reflection and synthesizing the final forensic verdict. |

## 5. RAG Integration & Evaluation (Section 4d)

ForensicSight incorporates a lightweight RAG system backed by a **Chroma vector store**. A domain-specific forensic corpus (sourced from digital forensics research papers, guides, and academic artifact evaluation criteria) was ingested.

- **Embedding Model:** HuggingFace `all-MiniLM-L6-v2`
- **Chunking Strategy:** Chunk size of 1000 with a 200 character overlap.

**Retrieval Evaluation:**
We verified context relevance empirically by evaluating the RAG subset against sample benchmark queries, ensuring accurate contextual grounding for the Analyst agent:

1. _"What are common artifacts in GAN generated faces?"_
2. _"How to detect AI lighting inconsistencies?"_
3. _"Identify signs of AI-generated hair and skin texturing."_
4. _"Detecting anomalous anatomical features in AI art."_

## 6. Setup & Installation Instructions

**Prerequisites:**

- Python 3.9+
- Git

**Installation:**

1. Clone the repository:
   ```bash
   git clone https://github.com/pulasthiav/AI-image-detector.git
   cd <REPOSITORY_NAME>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Environment Setup (Section 4f):
   Set up your environment variables by securely adding your Groq API Key to a .env file (for local use) or Streamlit Secrets (for cloud deployment).

4. Run the application locally:
   ```bash
   streamlit run app.py
   ```

## 7. Deployment & Live Demo Link (Section 4e)

- **Live Demo (Streamlit Community Cloud):** [Insert Live App URL Here]
- **2-Minute Screen-recorded Demo Link:** [Insert Demo Video URL Here]

## 8. Known Limitations & Future Work

- **Rate Limit Handling:** The system relies on third-party APIs (Groq), making it susceptible to API rate limits (TPM/TPD).
- **Map-Reduce Tiling Trade-offs:** Splitting one image into 4 quadrants exponentially increases the visual detail available but multiplies external API request quotas by four, generating higher loads on free-tier rate limits.

## 9. Tools & Libraries Disclosure (Section 10)

This project strictly builds upon the following toolset:

- **LangChain:** Agent architecture, routing, and message passing.
- **ChromaDB:** Local vector database for RAG context storage and retrieval.
- **Streamlit:** Fast, interactive front-end application framework.
- **Groq:** Ultra-low latency API provider for open-source Llama language and vision models.
