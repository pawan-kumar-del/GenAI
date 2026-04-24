# Technical Documentation

## 1. Introduction

### What is RAG
Retrieval-Augmented Generation (RAG) is a design pattern where an LLM first retrieves relevant knowledge from an external source and then uses that retrieved context to produce an answer. Instead of depending only on the model's parametric memory, the system grounds responses in documents that can be inspected and updated.

### Why It Is Needed
For customer support systems, raw LLM responses are risky because they may invent policies, unsupported product claims, or steps that do not exist in official documentation. RAG reduces that risk by:
- grounding answers in a known knowledge base
- improving factual consistency
- allowing content updates through document ingestion rather than model retraining
- enabling confidence-based escalation when the system is uncertain

### Use Case Overview
This project is a customer support assistant for a PDF knowledge base. It answers user questions through either:
- a CLI interface for testing and operations
- a Streamlit chat interface for interactive usage

The assistant retrieves relevant support content from a Chroma vector store, uses a Groq-hosted LLM to generate a grounded answer, and escalates to a human support path when the system lacks enough confidence.
In the Streamlit application, that escalation now creates a persistent local ticket that a human agent can resolve from an agent console.

## 2. System Architecture Explanation

### Detailed Explanation of HLD
The system has two main phases: ingestion and query-time inference.

In ingestion, the PDF file is loaded with `PyPDFLoader`, cleaned through text normalization, split into overlapping chunks, embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in ChromaDB.

In runtime inference, a user query enters the LangGraph workflow. The query is normalized first, then top-k relevant chunks are retrieved from Chroma. Those chunks are injected into a grounded prompt that is sent to the Groq LLM. After answer generation, a decision node determines whether the result should be delivered directly or routed to escalation.

When Streamlit receives an escalated result, it creates a HITL ticket containing the user query, escalation reason, confidence, and retrieved context. The same app exposes an `Agent View` where a human support agent can review the case and submit the final response.

### Component Interactions
- `main.py` and `streamlit_app.py` are entry points
- `service.py` coordinates ingestion and request execution
- `ingestion.py` handles PDF loading and chunk creation
- `vector_store.py` builds embeddings and the Chroma store
- `retriever.py` executes similarity search and ambiguity heuristics
- `llm_service.py` builds the prompt and invokes the LLM
- `workflow.py` defines the LangGraph state machine
- `hitl_store.py` persists escalation tickets and agent responses

The design is modular, so each concern is isolated and replaceable without rewriting the whole pipeline.

## 3. Design Decisions

### Chunk Size Choice
The current defaults are:
- chunk size: 1000 characters
- overlap: 200 characters

Why this is reasonable for the project:
- chunks are large enough to preserve procedural support context
- overlap reduces answer quality loss when important details are split across chunk boundaries
- the values keep total chunk count manageable for a single PDF

Trade-off:
- larger chunks improve coherence but may dilute retrieval precision
- smaller chunks improve pinpoint retrieval but may lose surrounding context

### Embedding Strategy
The project uses `all-MiniLM-L6-v2`, a compact sentence-transformer model. This is a pragmatic choice for an assignment because it is:
- lightweight
- widely used for semantic retrieval
- fast enough for local ingestion
- good enough for support-document similarity search

Embeddings are normalized, which aligns with the cosine similarity configuration used by Chroma.

### Retrieval Approach
The retriever uses top-k similarity search with relevance scores. This keeps the design straightforward and easy to reason about. The query is matched semantically against stored chunk embeddings, and the top results are forwarded to the LLM.

The implementation currently uses the maximum relevance score as the confidence value. That works as a simple routing signal, although it is less robust than using multiple-score aggregation, reranking, or answer verification.

### Prompt Design Logic
The prompt explicitly instructs the model to:
- answer only from retrieved context
- avoid guessing
- stay concise and actionable
- avoid fabricated claims

This prompt is intentionally strict because the application domain is customer support. The design favors safe escalation over speculative answers.

## 4. Workflow Explanation

### LangGraph Usage
LangGraph is used to model the runtime flow as a stateful graph rather than a single sequential chain. This matters because the system has routing behavior: some questions should end with an answer, while others should terminate in escalation.

### Node Responsibilities
- `input_processing_node`: validate and normalize the user query
- `retrieve_and_answer_node`: retrieve documents and invoke the LLM
- `decision_node`: inspect retrieval quality, ambiguity, and errors
- `output_node`: return direct answer
- `escalation_node`: return escalation response

### State Transitions
The state begins with the raw user query and empty runtime fields. Each node enriches the same state object. By the time the graph finishes, the state contains:
- cleaned query
- retrieved chunks
- similarity scores
- generated answer
- confidence
- escalation status
- escalation reason or error

This state-centric design makes the workflow inspectable and extensible.

The human response path is handled one layer above the graph. That is an intentional design choice: LangGraph is responsible for deciding whether automation is safe, and Streamlit is responsible for human queue handling and final response delivery.

## 5. Conditional Logic

### Intent Detection
There is no full classifier in the current project. Instead, the system uses lightweight heuristics to detect ambiguity. Queries such as `help` or very short prompts are treated as insufficiently specific.

### Routing Decisions
The routing logic is implemented in the LangGraph decision node. The answer is escalated when:
- the query is invalid
- retrieval returns no documents
- confidence is below threshold
- query appears ambiguous
- the LLM fails at runtime

Otherwise, the graph returns the LLM answer directly.

This is a practical first-pass routing design. It is easy to understand and fits the current project scale, even though it is not yet as strong as a dedicated intent classifier or confidence calibration pipeline.

## 6. HITL Implementation

### Role of Human Intervention
Human intervention is the safety fallback. When the system cannot confidently answer from retrieved content, it should stop and hand off the case rather than improvise. This is especially important in support contexts involving billing, account problems, policy edge cases, or missing documentation.

### Benefits
- reduces hallucination risk
- makes system behavior safer
- creates a clear boundary between automated and human support
- provides a path for handling unsupported or underspecified queries
- demonstrates a usable closed-loop support handoff inside the web app

### Limitations
The current project now includes a local human queue inside Streamlit, but it is still not a production helpdesk integration. Ticket storage is backed by a JSON file, which is fine for assignment-scale demos but not for concurrent or distributed deployments.

## 7. Challenges & Trade-offs

### Retrieval Accuracy vs Speed
- Higher `top_k` can improve context coverage but increases prompt size and latency
- Lower `top_k` is faster but may miss relevant support details

### Chunk Size vs Context Quality
- Large chunks preserve complete procedures but may reduce precision
- Small chunks increase retrieval specificity but risk fragmenting instructions

### Cost vs Performance
- Groq-hosted LLM inference gives good interactivity without local model hosting
- External APIs introduce dependency on network availability and provider limits
- Smaller embedding models are cheaper and faster, but stronger models may improve retrieval quality

### Simplicity vs Capability
The project intentionally uses a lean design:
- single document source
- one retriever strategy
- one routing threshold
- heuristic ambiguity detection

This keeps implementation simple, but leaves room for stronger production behavior.

## 8. Testing Strategy

### Testing Approach
The project currently supports practical functional testing through ingestion and end-to-end queries. A good testing strategy for this codebase should include:
- ingestion validation tests
- retrieval result tests
- workflow routing tests
- UI smoke tests for CLI and Streamlit
- failure-path tests for empty query and LLM exceptions
- HITL ticket creation, resolution, and customer delivery tests

### Sample Queries
- `How do I reset my password?`
- `I was charged twice, what should I do?`
- `help`
- `What discount can you give me today?`
- empty input

Expected outcomes:
- valid grounded support answers for supported topics
- escalation for vague or unsupported topics
- non-empty validation message for empty query
- creation of a pending ticket for escalated queries
- delivery of a human answer after agent resolution

### Recommended Additional Automated Tests
- unit tests for `normalize_text`
- unit tests for chunking validation rules
- tests for `is_query_ambiguous`
- graph tests asserting decision branches
- mock-based tests for LLM failures and empty retrieval scenarios

## 9. Future Enhancements

### Multi-Document Support
Extend ingestion to support multiple PDFs, metadata tagging, and collection-level filtering. This is the most natural next step because the current design is already modular.

### Feedback Loop
Capture whether answers were helpful, then use that signal to tune thresholding, chunking, or retrieval policies.

### Memory Integration
Introduce user/session memory for follow-up questions, while keeping support facts grounded in the vector store rather than conversational memory alone.

### Deployment
Convert the app into an API-backed service with:
- persistent service runtime
- reusable graph instances
- containerized deployment
- monitoring and structured logs
- database-backed ticketing integration for production HITL workflows

### Quality Improvements
- add reranking after initial retrieval
- move from heuristic ambiguity detection to intent classification
- use better confidence calibration
- add source citation formatting in responses
- support hybrid retrieval with keyword plus semantic search
- replace local JSON ticket storage with a real support backend
