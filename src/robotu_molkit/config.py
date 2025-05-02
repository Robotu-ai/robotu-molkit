# src/robotu_molkit/utils.py
import os
import json
from pathlib import Path
from typing import Optional, Tuple

CONFIG_PATH = Path.home() / ".config" / "molkit" / "config.json"

def load_credentials(
    override_api_key: Optional[str] = None,
    override_project_id: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    # 1) CLI override
    if override_api_key and override_project_id:
        return override_api_key, override_project_id
    # 2) Envvars
    api_key = os.getenv("IBM_API_KEY")
    project_id = os.getenv("IBM_PROJECT_ID")
    # 3) Config file
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        api_key = api_key or cfg.get("api_key")
        project_id = project_id or cfg.get("project_id")
    return api_key, project_id