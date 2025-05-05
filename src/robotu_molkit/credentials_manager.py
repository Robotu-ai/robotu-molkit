
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from robotu_molkit.constants import DEFAULT_WATSONX_AI_URL

class CredentialsManager:
    """
    Manage IBM Watsonx credentials and service URL: load, set, and retrieve API key,
    project ID, and Watsonx service URL from overrides, environment, or config file.
    """
    CONFIG_PATH = Path.home() / ".config" / "molkit" / "config.json"
    DEFAULT_URL = DEFAULT_WATSONX_AI_URL

    @classmethod
    def _ensure_dir(cls) -> None:
        cls.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _read_config(cls) -> Dict[str, Any]:
        if cls.CONFIG_PATH.exists():
            try:
                return json.loads(cls.CONFIG_PATH.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    @classmethod
    def _write_config(cls, cfg: Dict[str, Any]) -> None:
        cls._ensure_dir()
        cls.CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(
        cls,
        override_api_key: Optional[str] = None,
        override_project_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (api_key, project_id), checking in order:
          1) overrides,
          2) environment variables,
          3) config file.
        """
        # 1) CLI override
        if override_api_key and override_project_id:
            return override_api_key, override_project_id

        # 2) environment variables
        api_key = os.getenv("IBM_API_KEY")
        project_id = os.getenv("IBM_PROJECT_ID")

        # 3) config file
        cfg = cls._read_config()
        api_key = api_key or cfg.get("api_key")
        project_id = project_id or cfg.get("project_id")

        return api_key, project_id

    @classmethod
    def get_watsonx_url(cls) -> str:
        """
        Retrieve the Watsonx service URL from the environment or config, or use default.
        """
        # environment override
        url = os.getenv("WATSONX_URL")
        if url:
            return url
        # config file
        cfg = cls._read_config()
        return cfg.get("watsonx_url", cls.DEFAULT_URL)

    @classmethod
    def set_api_key(cls, api_key: str) -> None:
        """Save or update API key in config file."""
        cfg = cls._read_config()
        cfg["api_key"] = api_key
        cls._write_config(cfg)

    @classmethod
    def set_project_id(cls, project_id: str) -> None:
        """Save or update project ID in config file."""
        cfg = cls._read_config()
        cfg["project_id"] = project_id
        cls._write_config(cfg)

    @classmethod
    def set_watsonx_url(cls, url: str) -> None:
        """Save or update Watsonx service URL in config file."""
        cfg = cls._read_config()
        cfg["watsonx_url"] = url
        cls._write_config(cfg)