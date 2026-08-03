# ForensicSight: Agentic AI Image Forensics & Deepfake Detection System

## 1. Project Title & Overview

**Project Name:** ForensicSight: Agentic AI Image Forensics & Deepfake Detection System.

**Description:** An advanced agentic AI application designed to detect AI-generated images, digital manipulations, and deepfakes using a multi-agent pipeline, Chroma RAG knowledge base, and high-resolution map-reduce tiling. This project was developed to align strictly with the Horizon Campus IT41043 Agentic AI assignment requirements.

## 2. Architecture & Design Patterns (Section 4a)

ForensicSight leverages a robust multi-agent architecture implementing 3 distinct agentic patterns:

1. **Pattern 1: Router**
   - The Orchestrator agent receives the image uploaded by the user and routes it efficiently into the agentic forensic pipeline.
2. **Pattern 2: Tool-Use + Map-Reduce Tiling**
   - The Analyst agent queries a Chroma RAG index for domain-specific forensic guidelines.
   - The image is split into 4 high-resolution quadrants. The Analyst inspects each quadrant utilizing separate vision model calls (Map) and then aggregates the granular findings (Reduce).
3. **Pattern 3: Orchestrator-Worker & Reflection**
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

Our model strategy deliberately relies on a multi-provider setup to optimize latency, cost, and multimodal reasoning capabilities. 

| Sub-task | Model (Provider) | Why Chosen (Latency, Cost, Context, Reasoning) |
|---|---|---|
| Intent Routing | `llama-3.1-8b-instant` (Groq) | Ultra-low latency, near-zero cost, highly efficient for fast intent classification and routing decisions. |
| High-Res Vision Tiling Analysis | Groq Vision Model (4-tile Map-Reduce) | High-speed multimodal processing across image quadrants to detect structural visual artifacts. |
| Reporting & Synthesis | `llama-3.3-70b-versatile` (Groq) | Superior reasoning capacity and large context window required for deep reflection and synthesizing the final forensic verdict. |

## 5. RAG Integration & Evaluation (Section 4d)

ForensicSight incorporates a lightweight RAG system backed by a **Chroma vector store**. A domain-specific forensic corpus (comprising over 20+ documents and guidelines on identifying subtle AI artifacts) was ingested using a targeted chunking strategy. 
- **Retrieval Evaluation:** We evaluated the RAG system by passing sample queries related to "AI deepfake artifacts" and performed relevance checks on the retrieved chunks to ensure the Vision Analyst is grounded in highly relevant, factual criteria.

## 6. Setup & Installation Instructions

**Prerequisites:**
- Python 3.9+
- Git

**Installation:**
1. Clone the repository:
   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_NAME>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Environment Setup (Section 4f):
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```
   *(For Streamlit deployments, add these under the Streamlit Community Cloud Secrets management console).*

4. Run the application locally:
   ```bash
   streamlit run app.py
   ```

## 7. Deployment & Live Demo Link (Section 4e)

- **Live Demo (Streamlit Community Cloud):** [Insert Live App URL Here]

## 8. Known Limitations & Future Work

- **Rate Limit Handling:** The system relies on third-party APIs (Groq, OpenRouter), making it susceptible to API rate limits (TPM/TPD).
- **Map-Reduce Tiling Trade-offs:** Splitting one image into 4 quadrants exponentially increases the visual detail available but multiplies external API request quotas by four, generating higher loads on free-tier rate limits.
