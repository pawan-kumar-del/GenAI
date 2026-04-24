# Customer Support Assistant (RAG + LangGraph)

Production-like Retrieval-Augmented Generation assistant for customer support, built with:
- Python
- LangChain
- LangGraph
- ChromaDB
- Groq API
- PyPDF loader

## Project Structure

```text
CC_RAG/
├── .gitignore
├── .env.example
├── requirements.txt
├── README.md
├── ingest.py
├── main.py
├── streamlit_app.py
└── rag_support_assistant/
    ├── __init__.py
    ├── config.py
    ├── hitl_store.py
    ├── ingestion.py
    ├── llm_service.py
    ├── logging_utils.py
    ├── retriever.py
    ├── service.py
    ├── vector_store.py
    └── workflow.py
```

## Architecture

### Ingestion Pipeline
1. Load PDF via `PyPDFLoader`
2. Clean/normalize text
3. Split into configurable chunks
4. Embed chunks with HuggingFace embeddings
5. Persist vectors in ChromaDB

### Runtime Query Flow
1. User query enters LangGraph
2. Retrieval fetches top-k relevant chunks + relevance scores
3. LLM produces grounded answer using retrieved context only
4. Decision node checks confidence and ambiguity
5. Route to:
   - `output_node` (high confidence), or
   - `escalation_node` (HITL)
6. In Streamlit, escalated queries create a local HITL ticket for human review
7. A human agent resolves the ticket from the built-in agent console and the reply is delivered back to the customer chat

## LangGraph Workflow

Nodes:
- `input_processing_node`
- `retrieve_and_answer_node`
- `decision_node`
- `output_node`
- `escalation_node`

Edges:
- `START -> input_processing_node -> retrieve_and_answer_node -> decision_node`
- Conditional routing:
  - high confidence -> `output_node -> END`
  - low confidence / no docs / ambiguous -> `escalation_node -> END`

## Streamlit HITL Workflow

The Streamlit app now implements an end-to-end human-in-the-loop flow:
1. Customer asks a question in `Customer View`
2. LangGraph decides whether to answer directly or escalate
3. If escalated, the app creates a ticket in `data/hitl_tickets.json`
4. Agent switches to `Agent View`
5. Agent reviews:
   - customer question
   - escalation reason
   - retrieval confidence
   - retrieved context snippets
6. Agent submits a human response
7. Customer refreshes the chat and receives the human reply in the same conversation

## Setup

1. Create and activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment template and set API key:
   ```bash
   copy .env.example .env
   ```
4. Set:
   - `GROQ_API_KEY`
   - optional overrides like `CHUNK_SIZE`, `RETRIEVAL_TOP_K`, `SIMILARITY_THRESHOLD`
   - dependency versions from `requirements.txt` include `streamlit` for the web UI and `protobuf>=5.0,<7.0` for compatibility
5. Put your knowledge base PDF at:
   - `data/knowledge_base.pdf`
   - or pass a custom path during ingestion

## Usage

### 1) Ingest knowledge base
```bash
python ingest.py run --pdf-path data/knowledge_base.pdf
```

### 2) Single query
```bash
python main.py ask "How do I reset my account password?"
```

### 3) Interactive chat
```bash
python main.py chat
```

### 4) Streamlit web app
```bash
python -m streamlit run streamlit_app.py
```

The web app provides:
- chat-style question input
- answer, confidence, and escalation status
- retrieved source chunks for quick inspection
- PDF upload and live ingestion from the sidebar
- `Customer View` and `Agent View` in the same app
- local HITL ticket creation and resolution
- sidebar status for knowledge base and configuration

## Git Ignore

The project includes a `.gitignore` configured for:
- Python cache and build artifacts
- local virtual environments like `venv/`
- secret files such as `.env`
- local Chroma persistence in `chroma_db/`
- uploaded PDFs in `data/uploads/`
- local HITL ticket storage in `data/hitl_tickets.json`
- runtime log files including Streamlit logs

## System Prompt (RAG + HITL Constraints)

Defined in `rag_support_assistant/llm_service.py` as `SYSTEM_PROMPT`:
- Only answer from retrieved context
- Do not guess when context is insufficient
- Escalate to a human specialist when confidence is low or context is missing
- Avoid fabricated policy/product/legal claims

## Escalation Conditions (HITL)

The system escalates when:
1. No relevant documents are retrieved
2. Average relevance score is below `SIMILARITY_THRESHOLD`
3. Query is ambiguous based on heuristics
4. Input is invalid (empty query)

When running in Streamlit, an escalation now creates a persistent local ticket that can be handled by a human agent in `Agent View`.

## Error Handling

Implemented for:
- Missing/invalid PDF path
- Invalid chunking configuration
- Empty user query
- LLM/runtime API failures
- Retrieval failures and empty results
- HITL ticket resolution through local JSON storage

CLI commands return non-zero exit codes on unrecoverable failures.

## Sample Test Queries

Use these after ingesting your support PDF:

1. `How do I reset my password?`
   - Expected: grounded steps from policy/process context.
2. `I was charged twice, what should I do?`
   - Expected: billing workflow from docs, or escalation if missing.
3. `help`
   - Expected: ambiguous query detected -> escalation.
4. `What discount can you give me today?`
   - Expected: no fabrication; if unsupported in KB, escalation.
5. `` (empty input)
   - Expected: validation message asking for non-empty query.

## Scalability Notes

- Config-driven behavior via environment variables.
- Modular services allow replacing:
  - embedding model
  - vector store backend
  - LLM provider
  - decision policies
- Workflow graph can be extended with moderation, intent classification, external ticketing integration, and agent-assist tooling.

