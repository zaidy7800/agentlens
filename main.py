import ollama
from ddgs import DDGS

SYSTEM_PROMPT = """You are AgentLens, an expert AI assistant that recommends the best LLMs for agentic workflows.

You will be given a workflow description AND real web search results.
Use the search results to give accurate, up-to-date recommendations.

Respond in this exact format:

## Recommended LLMs

| Model | Provider | Parameters | Tool Calling | Best For |
|-------|----------|------------|--------------|----------|
| ...   | ...      | ...        | Yes/No       | ...      |

## Top Pick
**Model**: <name>
**Why**: <1-2 sentences>

## Key Considerations
- <point 1>
- <point 2>
- <point 3>

## Sources
- <source 1>
- <source 2>"""

EXAMPLES = [
    "marketing automation agent",
    "coding assistant agent",
    "research & summarization agent",
    "customer support agent",
    "RAG pipeline agent",
]


def get_models():
    try:
        return [m["model"] for m in ollama.list().get("models", [])]
    except:
        return []


def web_search(query: str) -> str:
    print(f"Searching web for: {query}")
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(f"best LLMs for {query} 2024", max_results=5):
                results.append(f"- {r['title']}: {r['body']} ({r['href']})")
    except Exception as e:
        print(f"Search failed: {e}")
        return "No search results available."
    return "\n".join(results) if results else "No results found."


def recommend(workflow: str) -> str:
    print("Researching on the web...")
    search_results = web_search(workflow)

    models = get_models()
    local_note = f"\nLocal models available: {', '.join(models)}" if models else ""

    user_prompt = f"""Workflow: {workflow}

Web Search Results:
{search_results}
{local_note}

Based on the search results above, recommend the best LLMs for this workflow."""

    print("Analyzing results...\n")
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def main():
    print("=" * 50)
    print("   AgentLens — LLM Research Agent")
    print("   Powered by DDGS + Ollama")
    print("=" * 50)

    models = get_models()
    print(f"\nLocal models: {', '.join(models) if models else 'none found'}")

    print("\nExamples:")
    for i, e in enumerate(EXAMPLES, 1):
        print(f"  {i}. {e}")
    print()

    while True:
        query = input("Enter workflow (or number, or 'quit'): ").strip()

        if query.lower() == "quit":
            break
        if not query:
            continue
        if query.isdigit() and 1 <= int(query) <= len(EXAMPLES):
            query = EXAMPLES[int(query) - 1]
            print(f"Selected: {query}\n")

        print("\nStarting research agent...\n")
        print(recommend(query))
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()