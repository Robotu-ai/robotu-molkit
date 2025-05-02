from __future__ import annotations
"""summary_generator.py
========================
**No hard‑coded prompts**
This module relies on plain‑text template files living under
``src/robotu_molkit/vector/prompts/`` and pristine backups in
``src/robotu_molkit/vector/prompts/default``.

If a user deletes or corrupts the editable copy they can always restore the
original via ``PromptManager.restore_default()``.
"""

import logging
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, List, Union

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from robotu_molkit.config import load_credentials
from robotu_molkit.constants import (
    DEFAULT_WATSONX_AI_URL,
    DEFAULT_WATSONX_GENERATIVE_MODEL,
)

__all__ = ["SummaryGenerator", "PromptManager"]


# ---------------------------------------------------------------------------
# Prompt manager – **no string literals for templates inside code** ----------
# ---------------------------------------------------------------------------
class PromptManager:
    """Load user‑editable prompt templates and their variable manifests.

    Directory layout (inside repo):

    ``vector/prompts/default``   ← pristine, version‑controlled templates
    ``vector/prompts``           ← editable copies (git‑ignored)

    Only filenames are referenced in code; the actual text is stored on disk.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or (Path(__file__).parent / "prompts")
        self.default_dir = self.base_dir / "default"
        # Ensure directories exist; *do not* create files from literals
        self.default_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # .......................................... public ........................
    def load_template(self, name: str) -> str:
        """Return the text in ``{name}_prompt.txt`` from the editable folder."""
        path = self.base_dir / f"{name}_prompt.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt '{name}' not found at {path}. Please create the file or restore default."
            )
        return path.read_text()

    def load_vars(self, name: str) -> Dict[str, str]:
        path = self.base_dir / f"{name}_vars.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Vars file for prompt '{name}' missing at {path}."
            )
        return json.loads(path.read_text())

    def restore_default(self, name: str) -> None:
        """Overwrite editable prompt/vars with the pristine default copies."""
        for suffix in ("prompt.txt", "vars.json"):
            src = self.default_dir / f"{name}_{suffix}"
            dst = self.base_dir / f"{name}_{suffix}"
            if not src.exists():
                raise FileNotFoundError(
                    f"Default file {src} is missing – repository is incomplete."
                )
            dst.write_text(src.read_text())


# ---------------------------------------------------------------------------
# Summary generator ----------------------------------------------------------
# ---------------------------------------------------------------------------
class SummaryGenerator:
    """Generate Granite summaries using external prompt templates."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        url: str = DEFAULT_WATSONX_AI_URL,
        model_id: str = DEFAULT_WATSONX_GENERATIVE_MODEL,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Resolve credentials from config if not explicitly supplied
        self.api_key, self.project_id = load_credentials(api_key, project_id)
        if not (self.api_key and self.project_id):
            raise ValueError(
                "Watsonx credentials missing – use `molkit config` or pass them explicitly."
            )

        # Generation parameters
        self.params = params or {
            GenParams.MAX_NEW_TOKENS: 120,
            GenParams.TEMPERATURE: 0.7,
        }

        creds = Credentials(api_key=self.api_key, url=url)
        self.model = ModelInference(
            model_id=model_id,
            params=self.params,
            credentials=creds,
            project_id=self.project_id,
        )

        self.prompts = PromptManager()

    # .......................................... public ......................
    def generate_general_summary(self, data: Dict[str, Any]) -> str:
        attrs = self._build_attrs(data)
        prompt_text = self.prompts.load_template("general").format(**attrs)
        try:
            rsp = self.model.generate_text(prompt=prompt_text)
        except Exception as exc:
            logging.warning("Granite generation error: %s", exc)
            return ""

        return self._unwrap_response(rsp).strip()

    # ................................ internal helpers ......................
    def _unwrap_response(self, rsp: Union[str, Dict[str, Any]]) -> str:
        """Normalise different ModelInference payload shapes."""
        if isinstance(rsp, str):
            return rsp
        if isinstance(rsp, dict):
            if "generated_text" in rsp:
                return rsp["generated_text"]
            if "text" in rsp:
                return rsp["text"]
            if rsp.get("results"):
                return rsp["results"][0].get("generated_text", "")
        return ""

    # ................................ attribute builders ....................
    def _build_attrs(self, data: Dict[str, Any]) -> Dict[str, str]:
        preferred = (
            data.get("names", {}).get("preferred")
            or data.get("names", {}).get("cas_like")
            or "Unknown"
        )

        ghs = data.get("safety", {}).get("ghs_codes", []) or []
        if any(c.startswith("H3") for c in ghs):
            hazard = "high hazard"
        elif any(c.startswith("H2") for c in ghs):
            hazard = "moderate hazard"
        elif ghs:
            hazard = "low hazard"
        else:
            hazard = "no known hazard"

        logp = data.get("solubility", {}).get("logp")
        if logp is None:
            sol = "unknown solubility"
        elif logp < -0.5:
            sol = "very soluble"
        elif logp <= 3:
            sol = "moderately soluble"
        else:
            sol = "poorly soluble"

        spectra_raw = data.get("spectra", {}).get("raw", {}) or {}
        keys = [k.replace(" Spectra", "") for k in spectra_raw.keys()]
        spectra_tag = ", ".join(keys) + " spectra available" if keys else "no spectra available"

        peak = self._extract_peak(spectra_raw)

        return {
            "preferred_name": preferred,
            "hazard_tag": hazard,
            "solubility_tag": sol,
            "spectra_tag": spectra_tag,
            "notable_peak": peak,
            "cid": str(data.get("search", {}).get("cid", "")),
        }

    # ........................................................................
    def _extract_peak(self, spectra_raw: Dict[str, Any]) -> str:
        for section in spectra_raw.values():
            if not isinstance(section, list):
                continue
            for item in section:
                val = item.get("Value", {})
                lines = val.get("StringWithMarkup") or []
                for seg in lines:
                    text = seg.get("String", "")
                    m = re.search(r"(\d+(?:\.\d+)?\s*(?:nm|m/z))", text, flags=re.I)
                    if m:
                        return m.group(1)
        return ""
