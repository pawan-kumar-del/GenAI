# High-Level Design (HLD)

## 1. System Overview

### Problem Definition
The project implements a Retrieval-Augmented Generation (RAG) based customer support assistant for answering questions from a PDF knowledge base. The core problem it solves is that a general-purpose LLM should not answer support queries from memory because that leads to hallucinations, inconsistent policy answers, and poor traceability. This system constrains answers using retrieved document chunks and routes uncertain cases to a human support path.

### Scope of the System
In scope:
- PDF-based knowledge base ingestion
- Text normalization and chunking
- Embedding generation for chunks
- Persistent vector indexing in ChromaDB
- Query-time retrieval and grounded answer generation
- LangGraph-driven workflow orchestration
- Escalation path for low-confidence or ambiguous queries
- Streamlit-based human agent console for resolving escalated tickets
- User interaction through CLI and Streamlit web UI

Out of scope in the current implementation:
- Multi-document corpus management
- External helpdesk or CRM integration
- User authentication and session persistence
- Feedback learning loop
- Deployment infrastructure

## 2. Architecture Diagram

```text
+----------------------------------------------------------------------------------+
|                            CUSTOMER SUPPORT ASSISTANT                            |
|                             (RAG + LangGraph + HITL)                             |
+----------------------------------------------------------------------------------+

  INGESTION FLOW
  ==============

  +------------------------+      +--------------------------+      +----------------------+
  | PDF Knowledge Base     | ---> | Document Ingestion       | ---> | Chunking Strategy    |
  | data/knowledge_base.pdf|      | PyPDFLoader + Cleaning   |      | Recursive Splitter   |
  +------------------------+      +--------------------------+      +----------------------+
                                                                                 |
                                                                                 v
                                                                  +--------------------------+
                                                                  | Embedding System         |
                                                                  | HuggingFaceEmbeddings    |
                                                                  | all-MiniLM-L6-v2         |
                                                                  +--------------------------+
                                                                                 |
                                                                                 v
                                                                  +--------------------------+
                                                                  | Vector Database          |
                                                                  | ChromaDB                |
                                                                  | Persistent Local Store   |
                                                                  +--------------------------+


  QUERY / ANSWER FLOW
  ===================

  +--------------------------+
  | User Interface           |
  | - CLI                    |
  | - Streamlit Web UI       |
  +--------------------------+
               |
               v
  +--------------------------+
  | Workflow Orchestration   |
  | LangGraph StateGraph     |
  +--------------------------+
               |
               v
  +--------------------------+
  | Input Processing Node    |
  | Query validation/cleanup |
  +--------------------------+
               |
               v
  +--------------------------+        +--------------------------+
  | Retrieval Layer          | -----> | Vector Database          |
  | Similarity search top-k  | <----- | ChromaDB                |
  +--------------------------+        +--------------------------+
               |
               v
  +--------------------------+
  | LLM Processing Layer     |
  | Groq Chat LLM            |
  | Grounded answer prompt   |
  +--------------------------+
               |
               v
  +--------------------------+
  | Decision / Routing Layer |
  | Confidence + ambiguity   |
  +--------------------------+
         |                               |
         | High confidence               | Low confidence / Missing context /
         v                               | Ambiguous query / LLM failure
  +--------------------------+           v
  | Output Node              |    +--------------------------+
  | Final automated answer   |    | HITL System              |
  +--------------------------+    | Escalation ticket store  |
         |                        | + Streamlit Agent View   |
         |                        +--------------------------+
         |                                   |
         v                                   v
  +--------------------------------------------------------------+
  | Final Response to User                                       |
  | - Direct RAG answer, or                                      |
  | - Human agent response returned into the same chat session   |
  +--------------------------------------------------------------+
```

## 3. Component Description

### Document Loader
- Implemented with `PyPDFLoader` in `rag_support_assistant/ingestion.py`
- Loads the PDF page by page into LangChain `Document` objects
- Preserves source metadata such as page number and source path

### Chunking Strategy
- Implemented with `RecursiveCharacterTextSplitter`
- Default configuration:
  - `chunk_size = 1000`
  - `chunk_overlap = 200`
- Separators prioritize paragraph and sentence boundaries before falling back to whitespace and character-level splitting
- Each chunk is tagged with `chunk_index` metadata

### Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2`
- Instantiated through `HuggingFaceEmbeddings`
- Embeddings are normalized to improve cosine similarity behavior

### Vector Store
- ChromaDB with local persistence under `./chroma_db`
- Collection metadata uses `hnsw:space = cosine`
- Stores chunk embeddings and associated metadata

### Retriever
- Uses `similarity_search_with_relevance_scores`
- Returns top `k` chunks and relevance scores
- Current confidence metric is the maximum retrieved similarity score, even though the wrapper property is named `average_similarity`

### LLM
- Groq-hosted chat model through `ChatGroq`
- Default model: `llama-3.3-70b-versatile`
- Prompt is explicitly grounded and forbids unsupported claims

### Graph Workflow Engine
- Implemented with LangGraph `StateGraph`
- Nodes:
  - `input_processing_node`
  - `retrieve_and_answer_node`
  - `decision_node`
  - `output_node`
  - `escalation_node`

### Routing Layer
- Decision logic checks:
  - invalid input
  - no retrieved documents
  - confidence below threshold
  - ambiguous query heuristic

### HITL Module
- LangGraph still decides when escalation is required
- Streamlit creates a persistent local ticket when escalation is triggered
- A human agent resolves the ticket in `Agent View`
- The resolved answer is delivered back into the original customer chat session
- Ticket persistence is implemented with `rag_support_assistant/hitl_store.py`

## 4. Data Flow

### Ingestion Flow: PDF to Vector Store
1. User runs ingestion through `python ingest.py run`.
2. The PDF is loaded page by page.
3. Text is normalized to collapse noisy whitespace.
4. Documents are split into overlapping chunks.
5. Each chunk is embedded using the configured sentence-transformer model.
6. Embeddings and metadata are stored in persistent ChromaDB.

### Query Lifecycle: User Question to Final Answer
1. User submits a question through CLI or Streamlit.
2. LangGraph initializes the workflow state.
3. `input_processing_node` cleans and validates the query.
4. `retrieve_and_answer_node` retrieves top-k chunks from Chroma.
5. The retrieved chunks are inserted into the LLM prompt.
6. The LLM produces a grounded answer or indicates insufficient context.
7. `decision_node` evaluates confidence and ambiguity.
8. Workflow routes to:
   - `output_node` for direct response, or
   - `escalation_node` for HITL handoff message.
9. If escalated in Streamlit, a ticket is created with question, reason, confidence, and retrieved context.
10. Human agent resolves the ticket from `Agent View`.
11. Final answer is delivered either from the LLM or from the human agent reply.

## 5. Technology Choices

### Why ChromaDB
- Lightweight and easy to run locally for assignment-scale systems
- Native LangChain integration
- Persistent local storage without separate database operations overhead
- Suitable for a single-PDF and small-to-medium corpus use case

### Why LangGraph
- Explicit workflow representation for query processing
- Clean separation of retrieval, answer generation, decisioning, and escalation
- Easier to extend than a single monolithic chain when more routing branches are introduced

### LLM Choice
- Groq-hosted `llama-3.3-70b-versatile`
- Good latency-performance tradeoff for interactive question answering
- Simple LangChain integration through `ChatGroq`
- External hosted model avoids running local inference infrastructure

### Additional Tools
- `PyPDFLoader` for PDF parsing
- `RecursiveCharacterTextSplitter` for chunk creation
- `sentence-transformers` for semantic embeddings
- `Typer` and `Rich` for CLI
- `Streamlit` for rapid web UI
- local JSON ticket storage for HITL queueing
- `pydantic-settings` for environment-driven configuration

## 6. Scalability Considerations

### Handling Large Documents
- Current implementation supports only one configured PDF path by default
- For larger documents, chunk count will grow linearly and increase embedding/indexing time
- Scaling path:
  - batch ingestion
  - metadata filtering
  - multiple collections or namespaces
  - incremental updates instead of full reingestion

### Increasing Query Load
- Query path rebuilds the vector store and workflow per request in CLI flow, which is acceptable for small load but inefficient at scale
- Streamlit partially improves this with `st.cache_resource`
- Scaling path:
  - reuse graph and retriever instances
  - run behind an API service
  - introduce concurrent request handling and connection pooling

### Latency Concerns
- Main latency contributors:
  - embedding computation during ingestion
  - vector similarity retrieval
  - external LLM API call
- Mitigations:
  - keep chunk sizes balanced
  - tune `top_k`
  - cache initialized services
  - optionally replace the embedding model or LLM with a faster variant for production
