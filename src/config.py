import os
from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODELS = {
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "key_var": "ANTHROPIC_API_KEY"},
    "haiku": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "key_var": "ANTHROPIC_API_KEY"},
    "gpt4o": {"provider": "openai", "model": "gpt-4o", "key_var": "OPENAI_API_KEY"},
    "gpt55": {"provider": "openai", "model": "gpt-5.5", "key_var": "OPENAI_API_KEY"},
    "gemini_flash": {"provider": "google", "model": "gemini-2.5-flash", "key_var": "GOOGLE_API_KEY"},
}
BUDGET_LEVELS = [2000, 1000, 500, 250, 125]
TASK_FAMILIES = ["optimization", "prediction", "translation"]
TASKS_PER_FAMILY = 3
RESULTS_DIR = "results"
def get_active_models():
    active = {}
    keys = {"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY, "OPENAI_API_KEY": OPENAI_API_KEY, "GOOGLE_API_KEY": GOOGLE_API_KEY}
    for name, cfg in MODELS.items():
        key = keys.get(cfg["key_var"], "")
        if key and not key.endswith("here") and len(key) > 10:
            active[name] = cfg
    return active
BUDGET_LEVELS = [2000, 1000, 500, 250, 125]
TASK_FAMILIES = ["optimization", "prediction", "translation"]
TASKS_PER_FAMILY = 3
RESULTS_DIR = "results"

def get_active_models():
    active = {}
    keys = {
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
    }
    for name, cfg in MODELS.items():
        key = keys.get(cfg["key_var"], "")
        if key and not key.endswith("here") and len(key) > 10:
            active[name] = cfg
    return active
