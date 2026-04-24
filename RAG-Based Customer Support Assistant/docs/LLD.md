# Low-Level Design (LLD)

## 1. Module-Level Design

### Document Processing Module
File: `rag_support_assistant/ingestion.py`

Responsibilities:
- Validate PDF existence
- Load PDF pages into LangChain `Document` objects
- Normalize page text before downstream processing

Key functions:
- `load_pdf_documents(pdf_path: Path) -> List[Document]`
- `normalize_text(raw_text: str) -> str`

### Chunking Module
File: `rag_support_assistant/ingestion.py`

Responsibilities:
- Split normalized documents into overlapping chunks
- Enforce valid chunk configuration
- Add `chunk_index` metadata

Key function:
- `split_documents_into_chunks(documents, chunk_size, chunk_overlap) -> List[Document]`

### Embedding Module
File: `rag_support_assistant/vector_store.py`

Responsibilities:
- Construct the sentence-transformer embedding model
- Normalize embeddings for cosine similarity

Key function:
- `build_embedding_model(config: AppConfig) -> HuggingFaceEmbeddings`

### Vector Storage Module
Files:
- `rag_support_assistant/vector_store.py`
- `rag_support_assistant/service.py`

Responsibilities:
- Create and configure Chroma collection
- Persist embedded chunks
- Return reusable vector store handle at runtime

Key functions:
- `build_vector_store(config, embedding_model) -> Chroma`
- `ingest_documents_into_vector_store(config, chunked_documents) -> Chroma`
- `get_vector_store(config) -> Chroma`

### Retrieval Module
File: `rag_support_assistant/retriever.py`

Responsibilities:
- Retrieve relevant chunks with scores
- Detect ambiguity through heuristics

Key functions:
- `retrieve_relevant_documents(vector_store, query, top_k) -> RetrievalResult`
- `is_query_ambiguous(cleaned_query: str) -> bool`

### Query Processing Module
Files:
- `rag_support_assistant/llm_service.py`
- `rag_support_assistant/workflow.py`

Responsibilities:
- Build grounded prompt
- Invoke LLM with retrieved context
- Generate answer text

Key functions:
- `build_llm(config) -> ChatGroq`
- `generate_grounded_answer(llm, user_query, retrieved_documents) -> str`

### Graph Execution Module
File: `rag_support_assistant/workflow.py`

Responsibilities:
- Define graph state
- Register nodes and edges
- Compile the workflow
- Route between direct answer and escalation branches

Key function:
- `build_support_assistant_graph(config, vector_store)`

### HITL Module
Files:
- `rag_support_assistant/workflow.py`
- `rag_support_assistant/hitl_store.py`
- `streamlit_app.py`

Responsibilities:
- Decide when escalation is required
- Persist escalation tickets
- Present pending tickets to agents
- Accept human responses and deliver them back to the customer chat

Key function:
- `escalation_node(state) -> Dict[str, Any]`
- `create_ticket(...) -> Dict[str, Any]`
- `list_tickets(...) -> List[Dict[str, Any]]`
- `resolve_ticket(...) -> Dict[str, Any]`

## 2. Data Structures

### Document Representation
LangChain `Document`

Relevant fields:
```python
Document(
    page_content: str,
    metadata: {
        "source": str,
        "page": int,
        "chunk_index": int
    }
)
```

### Chunk Format
Each chunk is a `Document` with:
- `page_content`: normalized chunk text
- `metadata.source`: original PDF path
- `metadata.page`: source page number
- `metadata.chunk_index`: chunk sequence index after splitting

### Embedding Structure
Produced internally by `HuggingFaceEmbeddings`

Logical form:
```python
{
    "text": str,
    "vector": List[float],
    "metadata": Dict[str, Any]
}
```

### Query-Response Schema
Used implicitly in CLI and Streamlit responses:

```python
{
    "answer": str,
    "confidence": float,
    "needs_escalation": bool,
    "escalation_reason": str,
    "retrieved_documents": List[Document],
    "similarity_scores": List[float],
    "error": str
}
```

### HITL Ticket Schema
Used by the Streamlit human escalation flow:

```python
{
    "ticket_id": str,
    "session_id": str,
    "user_query": str,
    "escalation_reason": str,
    "confidence": float,
    "retrieved_context": List[Dict[str, Any]],
    "status": "pending" | "resolved",
    "human_response": str,
    "resolved_by": str,
    "created_at": str,
    "resolved_at": str,
}
```

### State Object for Graph
Defined as `SupportAssistantState`:

```python
class SupportAssistantState(TypedDict):
    query: str
    cleaned_query: str
    retrieved_documents: List[Document]
    similarity_scores: List[float]
    answer: str
    confidence: float
    needs_escalation: bool
    escalation_reason: str
    error: str
```

## 3. Workflow Design (LangGraph)

### Nodes

#### Processing Node
`input_processing_node`

Responsibilities:
- Trim and normalize whitespace in query
- Validate non-empty input
- Set early escalation for invalid input

#### Retrieval + Answer Node
`retrieve_and_answer_node`

Responsibilities:
- Retrieve top-k documents
- Build LLM prompt from retrieved context
- Generate grounded answer
- Capture LLM runtime failures

#### Decision Node
`decision_node`

Responsibilities:
- Inspect retrieval result quality
- Evaluate ambiguity and errors
- Decide output or escalation path

#### Output Node
`output_node`

Responsibilities:
- Return final answer for direct-response cases

#### Escalation Node
`escalation_node`

Responsibilities:
- Convert escalation reason into user-facing message

### Edges

```text
START
  -> input_processing_node
  -> retrieve_and_answer_node
  -> decision_node
  -> conditional route
      -> output_node
      -> escalation_node
  -> END
```

### State Flow
- `query` enters graph from UI or CLI
- `cleaned_query` is produced by input processing
- `retrieved_documents` and `similarity_scores` are produced by retrieval
- `answer` is produced by LLM or escalation branch
- `confidence`, `needs_escalation`, `escalation_reason`, and `error` drive routing

## 4. Conditional Routing Logic

### Answer Generation Criteria
Direct answer is returned when:
- query is valid
- at least one document is retrieved
- confidence is above configured threshold
- query is not marked ambiguous
- no internal runtime error occurred

### Escalation Criteria

#### Low Confidence
Triggered when:
```python
confidence_score < config.similarity_threshold
```

Default threshold:
- `SIMILARITY_THRESHOLD = 0.30`

#### Missing Context
Triggered when:
- no documents are retrieved

#### Complex or Ambiguous Query
Triggered when:
- query is shorter than 4 tokens
- or it exactly matches one of the heuristic ambiguity markers:
  - `help`
  - `issue`
  - `problem`
  - `not working`
  - `what should i do`
  - `can you assist`

#### Runtime Failure
Triggered when:
- LLM invocation fails
- input query is empty

## 5. HITL Design

### When Escalation Is Triggered
Escalation happens when:
- user input is invalid
- retrieval yields no relevant chunks
- confidence score is low
- user query is ambiguous
- LLM service fails

### What Happens After Escalation
Current implementation behavior:
- workflow sets `needs_escalation = True`
- an `escalation_reason` is recorded
- `escalation_node` returns the human handoff message
- Streamlit creates a local ticket record in `data/hitl_tickets.json`
- the customer remains in the same chat session waiting for a human reply

### How Human Response Is Integrated
- `Agent View` lists all pending tickets
- the agent sees the original query, escalation reason, confidence, and retrieved snippets
- when the agent submits a response, the ticket status changes to `resolved`
- the customer session polls the store and injects the human response back into the chat history
- this is an application-level HITL loop and does not yet resume execution inside LangGraph itself

## 6. API / Interface Design

### Input Format

#### CLI
- Single query:
```bash
python main.py ask "How do I reset my password?"
```

#### Interactive CLI
```bash
python main.py chat
```

#### Streamlit
- Free-text input via `st.chat_input`
- Sidebar PDF upload for live ingestion
- `Customer View` for end users
- `Agent View` for human support agents

### Output Format

Returned state fields used externally:
- `answer`
- `confidence`
- `needs_escalation`
- retrieved source excerpts in Streamlit

CLI output format:
```text
Answer: <text>
confidence=<float>, escalated=<true|false>
```

Streamlit output format:
- answer text
- confidence badge
- escalation status
- expandable retrieved context list
- ticket identifiers for escalated requests
- human-agent responses once resolved

### Interaction Flow
1. User submits query
2. Runtime initializes graph state
3. Graph executes processing, retrieval, answer generation, and routing
4. If escalated, Streamlit creates a ticket
5. Agent resolves the ticket in `Agent View`
6. UI renders the resolved human response in the original customer conversation

## 7. Error Handling

### Missing Data
- Missing PDF path raises `FileNotFoundError`
- Empty query is trapped in `input_processing_node`

### No Relevant Chunks Found
- Retrieval returns empty list
- Decision node escalates with reason: no relevant documents found

### LLM Failure
- `retrieve_and_answer_node` catches exception
- Workflow stores error text
- User receives escalation message instead of raw stack trace

### Configuration Errors
- Invalid chunk settings raise `ValueError`
- Missing environment variables are caught by `AppConfig` validation

### Operational Limitation
- The vector store is assumed to exist at query time
- If ingestion has not been run, runtime initialization or retrieval may fail upstream
- HITL persistence is local JSON storage, so it is not suitable for multi-process or distributed production use without replacement
