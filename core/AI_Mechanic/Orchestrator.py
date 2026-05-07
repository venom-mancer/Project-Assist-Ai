import json
import base64
from pathlib import Path
from datetime import date
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from settings.json_reader import tavily_key
from settings.local_storage import VECTOR_DB_DIR, configure_local_storage

# 1. Initialize the Brain (Gemma 4 E4B)
llm = ChatOllama(model="gemma4:e4b")
configure_local_storage()

# 2. Initialize the Internet "Hands"
search = TavilySearchResults(k=3, api_key=tavily_key)
vector_db = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder=str(VECTOR_DB_DIR.parent / "model_cache" / "sentence_transformers"),
    ),
)
USER_PROFILE_PATH = Path("settings/user_profile.json")


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


def _load_user_profile():
    if not USER_PROFILE_PATH.exists():
        return {"learned_facts": []}
    return json.loads(USER_PROFILE_PATH.read_text(encoding="utf-8"))


def _save_user_profile(profile_data):
    USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_PROFILE_PATH.write_text(
        json.dumps(profile_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _extract_learning_fact(query, response_text):
    extraction_prompt = (
        "Extract one user learning fact from the following interaction.\n"
        "Return JSON only with keys: topic, skill_learned.\n"
        "If unclear, infer the broad topic and concise practical skill.\n\n"
        f"User query:\n{query}\n\n"
        f"Assistant response:\n{response_text}"
    )
    extraction = llm.invoke(extraction_prompt)
    raw_text = extraction.content if hasattr(extraction, "content") else str(extraction)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"topic": "General", "skill_learned": query.strip()[:160]}

    topic = str(parsed.get("topic", "General")).strip() or "General"
    skill = str(parsed.get("skill_learned", query)).strip() or query.strip()[:160]
    return {"topic": topic, "skill_learned": skill}


def _append_learned_fact(query, response_text):
    profile = _load_user_profile()
    learned_facts = profile.get("learned_facts", [])
    if not isinstance(learned_facts, list):
        learned_facts = []

    learned = _extract_learning_fact(query, response_text)
    learned_facts.append(
        {
            "date": date.today().isoformat(),
            "topic": learned["topic"],
            "skill_learned": learned["skill_learned"],
        }
    )
    profile["learned_facts"] = learned_facts
    _save_user_profile(profile)

def encode_image(image_path):
    """Turn an image into a base64 string for multimodal input."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _guess_mime_type(image_path):
    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"


def solve_real_world_problem(query, image_path=None):
    print(f"Assist-Eye is analyzing: {query}...")

    # Check local PDF library first, then web for live updates.
    local_docs = vector_db.similarity_search(query, k=2)
    library_context = "\n".join([doc.page_content for doc in local_docs])
    if not library_context.strip():
        library_context = "No local library matches found."
    web_data = search.invoke({"query": query})

    text_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {query}\n\n"
        f"Local Library Context:\n{library_context}\n\n"
        f"Web Data:\n{web_data}"
    )
    message_content = [{"type": "text", "text": text_content}]

    if image_path:
        mime_type = _guess_mime_type(image_path)
        base64_image = encode_image(image_path)
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

    message = HumanMessage(content=message_content)
    response = llm.invoke([message])
    response_text = response.content if hasattr(response, "content") else str(response)
    _append_learned_fact(query, response_text)
    return response_text


def solve_universal_problem(query, image_path=None):
    """Backward-compatible wrapper for older imports."""
    return solve_real_world_problem(query, image_path=image_path)