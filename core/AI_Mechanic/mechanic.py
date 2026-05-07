import json
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_community.tools.tavily_search import TavilySearchResults
from settings.json_reader import tavily_key

# 1. Initialize the Brain (Gemma 4 E4B)
llm = OllamaLLM(model="gemma4:e4b")

# 2. Initialize the Internet "Hands"
search = TavilySearchResults(k=3, api_key=tavily_key)


def build_system_prompt():
    prompt_path = Path("settings/prompt.json")
    prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))

    framework = prompt_data["operational_framework"]
    framework_steps = "\n".join(
        f"{index}. {step}" for index, step in enumerate(framework["steps"], start=1)
    )
    output_format = "\n".join(f"- {item}" for item in prompt_data["output_format"])
    safety_rules = "\n".join(
        f"- {item}" for item in prompt_data["critical_safety_guardrails"]
    )
    memory_rules = "\n".join(
        f"- {item}" for item in prompt_data["json_memory_protocol"]
    )

    return (
        f'### ROLE\n{prompt_data["role"]}\n\n'
        f'### OPERATIONAL FRAMEWORK ({framework["name"]})\n'
        f'{framework["instruction"]}\n{framework_steps}\n\n'
        f"### OUTPUT FORMAT\n{output_format}\n\n"
        f"### CRITICAL SAFETY GUARDRAILS\n{safety_rules}\n\n"
        f"### JSON & MEMORY PROTOCOL\n{memory_rules}"
    )


SYSTEM_PROMPT = build_system_prompt()

def solve_car_problem(query, image_path=None):
    print(f"Analyzing: {query}...")

    user_prompt = f"User Question: {query}\nImage Path: {image_path or 'None'}"

    # In a full LangGraph setup, the AI would call 'search' itself.
    # For now, let's trigger a search automatically for accuracy:
    web_data = search.invoke({"query": query})

    response = llm.invoke(
        f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nWeb Research Results: {web_data}"
    )
    return response