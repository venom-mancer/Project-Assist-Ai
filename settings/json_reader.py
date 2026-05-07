import json
from pathlib import Path


configs_path = Path("settings/configs.json")
with open(configs_path, "r", encoding="utf-8") as f:
    configs_data = json.load(f)

tavily_key = configs_data["tavily_config"]["tavily_key"]