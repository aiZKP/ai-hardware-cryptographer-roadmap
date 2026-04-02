# Module 5B — LLM Application Development

**Parent:** [Phase 3 — Artificial Intelligence](../../Guide.md) · Track B

> *Ship GenAI products — from prompt engineering to production deployment.*

**Prerequisites:** Module 3B (Agentic AI), Module 4B (ML Engineering).

**Role targets:** AI Engineer · GenAI Engineer · LLM Application Developer · Full-Stack AI Engineer

---

## Why This Matters for AI Hardware

LLM applications are the **largest consumer of GPU inference capacity** in 2025–2026:
- ChatGPT serves 200M+ weekly users → massive GPU fleet
- Enterprise RAG deployments → GPU-accelerated vector search + LLM inference
- Code assistants → long-context attention, streaming generation
- Understanding these patterns helps hardware engineers design chips that serve real demand

---

## 1. Prompt Engineering (Advanced)

* **System prompts:** persona, constraints, output format specification
* **Few-shot learning:** example selection, dynamic few-shot, chain-of-thought
* **Structured output:** JSON mode, function calling, tool use schemas
* **Prompt optimization:** iterative refinement, A/B testing prompts, automated evaluation
* **Long-context strategies:** context window management, chunking, summarization chains

---

## 2. Fine-Tuning LLMs

* **When to fine-tune vs prompt engineering vs RAG**
* **LoRA / QLoRA:** parameter-efficient fine-tuning, adapter merging
* **Full fine-tuning:** when you need maximum quality and have enough data/compute
* **Data preparation:** instruction formatting, chat templates, quality filtering
* **Evaluation:** perplexity, task-specific metrics, human eval, LLM-as-judge

**Projects:**
1. Fine-tune Llama-3-8B with QLoRA on a domain-specific Q&A dataset. Evaluate vs base model.
2. Merge LoRA adapters and export to ONNX for deployment.

---

## 3. Production RAG Architecture

* **Advanced retrieval:** hybrid search (dense + BM25), re-ranking (cross-encoder), query expansion
* **Chunking optimization:** recursive splitting, semantic chunking, parent-child retrieval
* **Multi-modal RAG:** images + text, document layout understanding
* **Evaluation framework:** RAGAS, context precision/recall, faithfulness scoring
* **Scaling:** distributed vector stores, caching, embedding batch processing

**Projects:**
1. Build a production RAG system with hybrid retrieval + re-ranking. Evaluate with RAGAS.
2. Add citation tracking — every generated claim linked to source chunks.

---

## 4. Production Deployment

* **API design:** streaming responses, structured output, error handling
* **Scaling patterns:** load balancing, auto-scaling, GPU right-sizing
* **Cost optimization:** caching (semantic cache, exact cache), model routing (small → large), prompt compression
* **Observability:** token usage tracking, latency monitoring, quality scoring
* **Safety:** content filtering, PII detection, output validation, rate limiting

**Projects:**
1. Deploy a RAG application with streaming, caching, and cost tracking. Measure tokens/$ efficiency.
2. Implement semantic caching — cache similar queries to reduce GPU inference calls by 30%+.

---

## Connection to Hardware

| Application pattern | Hardware implication |
|--------------------|---------------------|
| Long-context attention (128K tokens) | HBM bandwidth, KV-cache memory |
| Streaming token generation | Low-latency kernel scheduling |
| Batch inference serving | In-flight batching, GPU utilization |
| Vector search (RAG retrieval) | cuVS / FAISS on GPU |
| Multi-model routing | Multi-GPU scheduling, MIG partitioning |

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [Anthropic API Documentation](https://docs.anthropic.com/) | Claude API, tool use, streaming |
| [OpenAI Cookbook](https://cookbook.openai.com/) | GPT API patterns and best practices |
| [LlamaIndex](https://docs.llamaindex.ai/) | RAG framework |
| [RAGAS](https://docs.ragas.io/) | RAG evaluation framework |
| *Building LLM Applications* (various) | End-to-end LLM app development |
