import os
from langchain_ollama import OllamaLLM
from langchain_community.tools.tavily_search import TavilySearchResults
from settings.json_reader import tavily_key

# 1. Initialize the Brain (Gemma 4 E4B)
llm = OllamaLLM(model="gemma4:e4b")

# 2. Initialize the Internet "Hands"
search = TavilySearchResults(k=3, api_key=tavily_key)

def solve_car_problem(query, image_path=None):
    print(f"Analyzing: {query}...")
    
    # Simple Agentic Logic
    prompt = f"""
    You are a Master Mechanic AI. 
    User Question: {query}
    
    Step 1: Analyze the visual symptoms (if image provided).
    Step 2: Search web for technical bulletins or forum solutions.
    Step 3: Combine info into a clear repair guide.
    """
    
    # In a full LangGraph setup, the AI would call 'search' itself.
    # For now, let's trigger a search automatically for accuracy:
    web_data = search.invoke({"query": query})
    
    response = llm.invoke(f"{prompt}\n\nWeb Research Results: {web_data}")
    return response

# Example Usage
# result = solve_car_problem("How to fix squeaking brakes on a 2024 Honda Civic?")
# print(result)