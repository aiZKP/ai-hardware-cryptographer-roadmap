# Lecture 07 — Claude Agent SDK

**Track B · Agentic AI & GenAI** | [← Lecture 06](Lecture-06.md) | [Next →](Lecture-08.md)

---

## Learning Objectives

- Use the Claude Agent SDK to build production-grade agents
- Implement subagents for parallelism and isolation
- Handle streaming in agent loops
- Use computer use for browser/GUI automation

---

## 1. SDK Overview

The Claude Agent SDK (`claude_agent_sdk` / `anthropic` package) provides higher-level abstractions over the raw Messages API for building production agents.

```bash
pip install anthropic
```

**Key SDK concepts:**

| Concept | What it is |
|---------|-----------|
| `Anthropic` client | Base API client |
| `messages.create` | Core inference call |
| `messages.stream` | Streaming inference |
| Tool definitions | Structured function schemas |
| Computer use | Screen/GUI interaction beta |

---

## 2. Production Agent with SDK

A well-structured agent class wrapping the SDK:

```python
import anthropic
import json
import logging
from typing import Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    max_iterations: int = 30
    temperature: float = 0.0
    system: str = "You are a helpful AI assistant."

@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    iterations: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        pricing = {
            "claude-opus-4-6":           (15.00, 75.00),
            "claude-sonnet-4-6":         (3.00,  15.00),
            "claude-haiku-4-5-20251001": (0.80,   4.00),
        }
        # default to sonnet pricing
        inp, out = pricing.get("claude-sonnet-4-6", (3.00, 15.00))
        return (self.total_input_tokens * inp + self.total_output_tokens * out) / 1_000_000


class ClaudeAgent:
    """Production Claude agent with tool use, logging, and cost tracking."""

    def __init__(
        self,
        config: AgentConfig = None,
        tools: list[dict] = None,
        tool_handlers: dict[str, Callable] = None
    ):
        self.config = config or AgentConfig()
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.client = anthropic.Anthropic()

    def add_tool(self, schema: dict, handler: Callable):
        """Register a tool with its handler."""
        self.tools.append(schema)
        self.tool_handlers[schema["name"]] = handler

    def run(self, task: str, messages: list = None) -> AgentResult:
        """Run the agent on a task."""
        messages = messages or [{"role": "user", "content": task}]
        result = AgentResult(answer="")
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}")

            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "system": self.config.system,
                "messages": messages,
            }
            if self.tools:
                kwargs["tools"] = self.tools

            response = self.client.messages.create(**kwargs)

            # Track usage
            result.total_input_tokens += response.usage.input_tokens
            result.total_output_tokens += response.usage.output_tokens
            result.iterations = iteration

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                result.answer = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                break

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    logger.info(f"Tool call: {block.name}({block.input})")
                    result.tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                        "iteration": iteration
                    })

                    if block.name in self.tool_handlers:
                        try:
                            output = self.tool_handlers[block.name](**block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(output) if not isinstance(output, str) else output
                            })
                        except Exception as e:
                            logger.error(f"Tool {block.name} failed: {e}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Error: {e}",
                                "is_error": True
                            })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Unknown tool: {block.name}",
                            "is_error": True
                        })

                messages.append({"role": "user", "content": tool_results})

            else:
                logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
                break

        logger.info(
            f"Done in {iteration} iterations, "
            f"{result.total_input_tokens}→{result.total_output_tokens} tokens, "
            f"~${result.estimated_cost_usd:.4f}"
        )
        return result


# Usage
def search(query: str) -> str:
    return f"Results for: {query}"  # mock

def get_file(path: str) -> str:
    with open(path) as f:
        return f.read()

agent = ClaudeAgent(
    config=AgentConfig(model="claude-sonnet-4-6", max_iterations=10),
    tools=[
        {
            "name": "search",
            "description": "Search the web for information.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    ],
    tool_handlers={"search": search}
)

result = agent.run("What is the current state of the art in ML compiler technology?")
print(result.answer)
print(f"Cost: ${result.estimated_cost_usd:.4f}")
```

---

## 3. Subagents for Parallelism

Launch specialized subagents and aggregate their results:

```python
import asyncio
import anthropic

client = anthropic.Anthropic()

def run_subagent(role: str, task: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """Run a specialized subagent synchronously."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=f"You are a specialized {role}. Be concise and accurate.",
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text

async def run_subagent_async(role: str, task: str) -> tuple[str, str]:
    """Async wrapper for parallel execution."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_subagent, role, task)
    return role, result

async def orchestrate_research(topic: str) -> str:
    """Orchestrator + specialized subagents pattern."""

    # Run 3 research subagents in parallel
    tasks = [
        run_subagent_async("hardware expert", f"Explain the hardware implications of {topic}"),
        run_subagent_async("software expert", f"Explain the software stack for {topic}"),
        run_subagent_async("market analyst", f"Describe the market landscape for {topic}"),
    ]

    results = await asyncio.gather(*tasks)
    research = {role: result for role, result in results}

    # Orchestrator synthesizes
    synthesis_prompt = "\n\n".join(
        f"## {role.title()}\n{result}"
        for role, result in research.items()
    )

    final = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"Synthesize this research on {topic} into a coherent summary:\n\n{synthesis_prompt}"
        }]
    )
    return final.content[0].text

# Run
result = asyncio.run(orchestrate_research("MLIR for custom AI accelerators"))
print(result)
```

---

## 4. Streaming in Agent Loops

Show progress to users while the agent runs long tasks:

```python
import anthropic
from typing import Generator

client = anthropic.Anthropic()

def streaming_agent(task: str) -> Generator[str, None, None]:
    """Agent that yields text tokens as they arrive."""
    messages = [{"role": "user", "content": task}]

    while True:
        # Collect full response with streaming visible to user
        collected_content = []

        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text  # stream to caller
                # (we still need the full response for tool use)

            final = stream.get_final_message()

        messages.append({"role": "assistant", "content": final.content})

        if final.stop_reason == "end_turn":
            break

        elif final.stop_reason == "tool_use":
            yield "\n[executing tools...]\n"
            tool_results = []
            for block in final.content:
                if block.type == "tool_use":
                    result = TOOL_MAP.get(block.name, lambda **_: "Unknown tool")(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            messages.append({"role": "user", "content": tool_results})

# Usage in a CLI
for chunk in streaming_agent("Analyze the bandwidth requirements for training a 70B parameter model"):
    print(chunk, end="", flush=True)
```

---

## 5. Prompt Caching (Cost Optimization)

For agents that use the same large system prompt or documents repeatedly:

```python
# Mark large static content for caching
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an AI hardware engineer assistant.",
        },
        {
            "type": "text",
            "text": open("hardware_reference_manual.txt").read(),  # large doc
            "cache_control": {"type": "ephemeral"}  # cache this for 5 minutes
        }
    ],
    messages=[{"role": "user", "content": "What is the HBM3 bandwidth spec?"}]
)

# Subsequent calls with same cached content are ~90% cheaper
# Cache hit shows in: response.usage.cache_read_input_tokens
print(f"Cache read: {response.usage.cache_read_input_tokens}")
print(f"Cache write: {response.usage.cache_creation_input_tokens}")
```

---

## Key Takeaways

1. Wrap the raw SDK in a `ClaudeAgent` class for reuse: config, logging, cost tracking
2. Subagents = specialized single-purpose agents run in parallel for speed
3. Streaming is async text generation — still need the final message for tool-use detection
4. Prompt caching dramatically reduces cost for agents with large static context (docs, manuals)
5. Always track `input_tokens` + `output_tokens` per run — costs compound across agent loops

---

## Exercises

1. Extend `ClaudeAgent` with a budget cap — raise `BudgetExceededError` if cost exceeds a threshold.
2. Build a 3-subagent research system where each subagent has different tools (web search, code exec, database).
3. Implement prompt caching for an agent that answers questions from a large codebase — measure the token savings.

---

**Previous:** [Lecture 06](Lecture-06.md) | **Next:** [Lecture 08 — Multi-Agent Systems](Lecture-08.md)
