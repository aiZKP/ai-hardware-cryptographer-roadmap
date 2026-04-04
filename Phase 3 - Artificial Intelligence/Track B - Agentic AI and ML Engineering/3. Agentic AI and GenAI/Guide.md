# Module 3B — Agentic AI & GenAI

**Parent:** [Phase 3 — Artificial Intelligence](../../Guide.md) · Track B

> *Build applications on top of large language models — agents, RAG, tool use, GenAI products.*

**Prerequisites:** Module 1 (Neural Networks), Module 2 (Frameworks — understand transformers and PyTorch).

**Role targets:** Agentic AI Engineer · GenAI Engineer · AI Engineer

---

## Lecture Series

12 lectures + 3 hands-on labs covering everything from LLM fundamentals to production multi-agent systems. **[→ Start here](Lectures/README.md)**

| # | Lecture | # | Lecture |
|---|---------|---|---------|
| [01](Lectures/Lecture-01.md) | LLM Fundamentals | [07](Lectures/Lecture-07.md) | Claude Agent SDK |
| [02](Lectures/Lecture-02.md) | Prompt Engineering | [08](Lectures/Lecture-08.md) | Multi-Agent Systems |
| [03](Lectures/Lecture-03.md) | Tool Use | [09](Lectures/Lecture-09.md) | RAG — Ingestion |
| [04](Lectures/Lecture-04.md) | Agent Architecture | [10](Lectures/Lecture-10.md) | RAG — Retrieval |
| [05](Lectures/Lecture-05.md) | Memory Systems | [11](Lectures/Lecture-11.md) | Evaluation |
| [06](Lectures/Lecture-06.md) | LangGraph | [12](Lectures/Lecture-12.md) | Production |

**Labs:** [Lab 01 — Research Agent](Lectures/Lab-01-Research-Agent.md) · [Lab 02 — Multi-Agent Pipeline](Lectures/Lab-02-Multi-Agent-Pipeline.md) · [Lab 03 — Production RAG](Lectures/Lab-03-Production-RAG.md)

---

## Why This Matters for AI Hardware

Agentic AI creates the **inference demand** that drives chip design:
- Long-context attention (128K+ tokens) → L5: HBM bandwidth, memory hierarchy
- Multi-turn tool calling → L3: low-latency kernel launch, stream scheduling
- Batch inference serving → L2: TensorRT-LLM, in-flight batching optimization
- RAG vector search → L1: cuVS/FAISS acceleration on GPU

Understanding these workloads helps L2 (compiler) and L5 (architecture) engineers design for real usage patterns.

---

## 1. LLM Fundamentals for Engineers

* **Transformer architecture:** attention mechanism, KV-cache, positional encoding
* **Tokenization:** BPE, SentencePiece, vocabulary size impact on embedding layer
* **Inference mechanics:** prefill (compute-bound) vs decode (memory-bound), autoregressive generation
* **Scaling laws:** parameter count vs dataset size vs compute budget

---

## 2. Agentic AI

* **What agents are:** LLM + tools + memory + planning loop
* **Agent frameworks:** LangChain, LangGraph, CrewAI, AutoGen, Claude Agent SDK
* **Tool use:** function calling, API integration, code execution
* **Memory:** conversation history, vector store retrieval, working memory
* **Planning:** chain-of-thought, ReAct, tree-of-thought, self-reflection
* **Multi-agent systems:** task decomposition, agent collaboration, orchestration

**Projects:**
1. Build an agent that uses tools (web search, calculator, code execution) to answer complex questions.
2. Build a multi-step research agent: given a topic, search → synthesize → write a report.

---

## 3. RAG (Retrieval-Augmented Generation)

* **Architecture:** document ingestion → chunking → embedding → vector store → retrieval → generation
* **Embedding models:** sentence-transformers, OpenAI embeddings, Cohere
* **Vector stores:** FAISS, Chroma, Pinecone, Weaviate, Milvus
* **Chunking strategies:** fixed-size, recursive, semantic, document-structure-aware
* **Retrieval:** similarity search, hybrid (dense + sparse), re-ranking
* **Evaluation:** faithfulness, relevance, hallucination detection

**Projects:**
1. Build a RAG pipeline over a technical documentation corpus. Evaluate retrieval quality.
2. Compare FAISS (CPU) vs FAISS (GPU) vs cuVS for vector search latency at 1M documents.

---

## 4. GenAI Product Development

* **Prompt engineering:** system prompts, few-shot, chain-of-thought, output formatting
* **Fine-tuning:** LoRA, QLoRA, full fine-tuning on domain data
* **Evaluation:** automated metrics (ROUGE, BLEU), LLM-as-judge, human evaluation
* **Guardrails:** output filtering, content safety, hallucination mitigation
* **Production deployment:** API design, streaming, rate limiting, cost management

**Projects:**
1. Fine-tune a 7B model with QLoRA on a domain-specific dataset. Measure improvement vs base model.
2. Deploy a GenAI application with streaming, safety guardrails, and cost tracking.

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [LangChain Documentation](https://python.langchain.com/) | Agent and RAG framework |
| [Claude Agent SDK](https://github.com/anthropics/claude-code) | Building agents with Claude |
| [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) | Practical Claude API examples |
| [RAG best practices](https://docs.llamaindex.ai/) | LlamaIndex documentation |
| *Build a Large Language Model (From Scratch)* (Raschka) | LLM internals |

---

## Next

→ [**Module 4B — ML Engineering & MLOps**](../4.%20ML%20Engineering%20and%20MLOps/Guide.md)
