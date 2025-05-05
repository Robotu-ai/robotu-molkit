# -*- coding: utf-8 -*-
"""summary_generator.py – Generate Granite summaries from external prompt files.
• **Safer response parsing**  Granite sometimes prepends or appends noise. We now:
  1. Instruct Granite with a valid JSON block (`"summary": "…"`).
  2. Extract the *first* JSON object containing the key ``summary`` and parse it
     with ``json.loads`` to obtain the value.
  3. Fall back to a regex/glance if JSON is not valid.

• ``_RESPONSE_BLOCK`` switched to canonical JSON with double quotes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
# Prompt manager ------------------------------------------------------------
# ---------------------------------------------------------------------------
class PromptManager:
    """Load user‑editable prompt templates and their variable manifests."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or (Path(__file__).parent / "prompts")
        self.default_dir = self.base_dir / "default"
        self.default_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_template(self, name: str) -> str:
        path = self.base_dir / f"{name}_prompt.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt '{name}' not found at {path}. Create the file or restore default."
            )
        return path.read_text()

# ---------------------------------------------------------------------------
# Summary generator ----------------------------------------------------------
# ---------------------------------------------------------------------------
class SummaryGenerator:
    """Generate Granite summaries (general & thematic)."""

    SUMMARY_TYPES: List[str] = ["general", "pharma", "materials", "safety", "spectra"]

    # Force Granite to respond with clean JSON
    _RESPONSE_BLOCK = (
        "\n\nReturn only this response Format:\n{\n  \"summary\": \"{response}\"\n}"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        url: str = DEFAULT_WATSONX_AI_URL,
        model_id: str = DEFAULT_WATSONX_GENERATIVE_MODEL,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.api_key, self.project_id = load_credentials(api_key, project_id)
        if not (self.api_key and self.project_id):
            raise ValueError("Missing Watsonx credentials; run `molkit config` or supply them.")

        creds = Credentials(api_key=self.api_key, url=url)
        self.model = ModelInference(
            model_id=model_id,
            params=params or {GenParams.MAX_NEW_TOKENS: 1200, GenParams.TEMPERATURE: 0.2},
            credentials=creds,
            project_id=self.project_id,
        )
        self.prompts = PromptManager()

    # ------------------------------------------------------------------
    # Public API --- calling Granite model to generate summary 
    # ------------------------------------------------------------------
    def generate_summary(self, data: Dict[str, Any], summary_type: str = "general") -> str:
        if summary_type not in self.SUMMARY_TYPES:
            raise ValueError(f"Unknown summary_type '{summary_type}'.")

        attrs = self._build_attrs(data)
        base_prompt = self.prompts.load_template(summary_type).format(**attrs)
        prompt_text = base_prompt + self._RESPONSE_BLOCK

        try:
            rsp = self.model.generate_text(prompt=prompt_text)
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Granite generation error (%s): %s", summary_type, exc)
            return ""

        # Append metadata block to the model output
        summary_text = self._extract_json_summary(rsp)
        metadata = attrs.get("metadata_block", "")

        return summary_text.strip() + "\n\n" + metadata.strip() if metadata else summary_text.strip()


    def generate_general_summary(self, data: Dict[str, Any]) -> str:  # noqa: D401
        return self.generate_summary(data, "general")

    def generate_all_summaries(self, data: Dict[str, Any]) -> Dict[str, str]:
        return {t: self.generate_summary(data, t) for t in self.SUMMARY_TYPES}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_json_summary(self, txt: str) -> str:
        """Extract summary value from the first JSON object containing the key `summary`."""
        # Locate the first brace that starts a JSON object
        json_match = re.search(r"\{[^{}]*?\"summary\"[^{}]*?\}", txt, flags=re.S)
        if json_match:
            block = json_match.group(0)
            try:
                obj = json.loads(block)
                if isinstance(obj, dict) and "summary" in obj:
                    return obj["summary"].strip()
            except json.JSONDecodeError:
                # attempt to salvage single‑quoted JSON
                try:
                    obj = json.loads(block.replace("'", '"'))
                    return obj.get("summary", "").strip()
                except Exception:  # noqa: BLE001
                    pass
        # Fallback – grab the first line, in case JSON failed
        return txt.split("\n", 1)[0].strip()

    # ---------------------------------------------------------------------
    # Utilidades de clasificación cualitativa
    # ---------------------------------------------------------------------
    @staticmethod
    def _qualitative_hazard(ghs: List[str]) -> str:
        """Devuelve una descripción cualitativa del peligro GHS."""
        if not ghs:
            return "no known hazard"

        hmap = {
            # Toxicidad aguda
            "H300": "fatal if swallowed",
            "H301": "toxic if swallowed",
            "H302": "harmful if swallowed",
            "H303": "may be harmful if swallowed",
            "H304": "may be fatal if swallowed and enters airways",
            "H305": "may be harmful if swallowed and enters airways",
            "H310": "fatal in contact with skin",
            "H311": "toxic in contact with skin",
            "H312": "harmful in contact with skin",
            "H313": "may be harmful in contact with skin",
            "H330": "fatal if inhaled",
            "H331": "toxic if inhaled",
            "H332": "harmful if inhaled",
            "H333": "may be harmful if inhaled",
            # Corrosión / irritación / ojos / piel
            "H314": "causes severe skin burns and eye damage",
            "H315": "causes skin irritation",
            "H316": "causes mild skin irritation",
            "H317": "may cause an allergic skin reaction",
            "H318": "causes serious eye damage",
            "H319": "causes serious eye irritation",
            # Sensibilización / STOT / SNC
            "H334": "may cause allergy or asthma symptoms or breathing difficulties if inhaled",
            "H335": "may cause respiratory irritation",
            "H336": "may cause drowsiness or dizziness",
            # Mutagenicidad, carcinogenicidad, repro‑tox
            "H340": "may cause genetic defects",
            "H341": "suspected of causing genetic defects",
            "H350": "may cause cancer",
            "H351": "suspected of causing cancer",
            "H360": "may damage fertility or the unborn child",
            "H361": "suspected of damaging fertility or the unborn child",
            # Toxicidad específica de órganos
            "H370": "causes damage to organs",
            "H371": "may cause damage to organs",
            "H372": "causes damage to organs through prolonged or repeated exposure",
            "H373": "may cause damage to organs through prolonged or repeated exposure",
        }

        hazards = {hmap[c] for c in ghs if c in hmap}
        tag = ", ".join(sorted(hazards))

        # Selección simple de pictograma GHS
        if any(code in ("H300", "H310", "H330") for code in ghs):
            picto = "skull-crossbones"
        elif any(code.startswith(("H34", "H35", "H36", "H37")) or code in ("H304",) for code in ghs):
            picto = "health-hazard"
        else:
            picto = "exclamation"

        return f"{tag}; pictogram: {picto}" if picto else tag

    @staticmethod
    def _qualitative_sol(logs: Optional[float]) -> str:
        if logs is None:
            return "unknown solubility"
        if logs > -0.5:
            return "very soluble"
        elif logs > -1.5:
            return "soluble"
        elif logs > -3.0:
            return "moderately soluble"
        elif logs > -4.0:
            return "sparingly soluble"
        else:
            return "insoluble"

    def format_spectra_info(self, spectra):
        spectra_keys = [k.replace(" Spectra", "") for k in spectra.keys()]
        spectra_tag = (
            ", ".join(spectra_keys) + " spectra available"
            if spectra_keys
            else "no spectra available"
        )
        notable_peak = self._extract_peak(spectra)
        return spectra_tag, notable_peak

    # ---------------------------------------------------------------------
    # Construcción de atributos para el prompt de Granite
    # ---------------------------------------------------------------------
    def _build_attrs(self, data: Dict[str, Any]) -> Dict[str, str]:
        from typing import Optional, Any, Dict
        import re

        names   = data.get("names", {})
        safety  = data.get("safety", {})
        sol     = data.get("solubility", {})
        spectra = data.get("spectra", {}).get("raw", {}) or {}
        meta    = data.get("meta", {})
        search  = data.get("search", {})

        preferred = (
            names.get("preferred")
            or names.get("cas_like")
            or names.get("systematic")
            or "Unknown"
        )
        cid = str(search.get("cid", ""))

        # Hazard tag
        hazard_tag = self._qualitative_hazard(safety.get("ghs_codes", []))

        # Qualitative solubility
        logs_val: Optional[float] = sol.get("logs")
        if logs_val is None and (lp := sol.get("logp")) is not None:
            logs_val = -0.4 * lp - 0.23
        solubility_tag = self._qualitative_sol(logs_val)

        # Numeric logP string
        logp_val = sol.get("logp")
        logp_str = f"{logp_val:.2f}" if isinstance(logp_val, (int, float)) else "n.a."

        # Spectra info
        spectra_tag, notable_peak = self.format_spectra_info(spectra)

        # Alias tag
        synonyms = [
            s for s in names.get("synonyms", [])
            if not re.fullmatch(r"\d{2,7}-\d{2}-\d", s)
            and s.lower() != preferred.lower()
        ]
        unique_syn = list(dict.fromkeys(synonyms))
        alias_tag = ", ".join(unique_syn[:10]) if unique_syn else (names.get("cas_like") or "n.a.")

        # Chemical ontology tag
        onto = meta.get("ontology", [])
        chem_tag = ", ".join(onto[:5]) if onto else "unclassified compound"

        # Extracted fields
        smiles = search.get("smiles", "n.a.")
        formula = search.get("formula", "n.a.")
        molecular_weight = str(search.get("molecular_weight", "n.a."))
        ghs_codes = ", ".join(safety.get("ghs_codes", []))
        ring_count = str(search.get("ring_count", "n.a."))
        aromatic_ring_count = str(search.get("aromatic_ring_count", "n.a."))
        rotatable_bonds = str(search.get("rotatable_bonds", "n.a."))

        metadata_block = f"""
            SMILES: {smiles}
            Formula: {formula}
            MW: {molecular_weight}
            logP: {logp_str}
            GHS codes: {ghs_codes}
            Rings: {ring_count}, Aromatic: {aromatic_ring_count}, Rotatable: {rotatable_bonds}
            """.strip()

        return {
            "preferred_name": preferred,
            "cid": cid,
            "hazard_tag": hazard_tag,
            "solubility_tag": solubility_tag,
            "logp": logp_str,
            "spectra_tag": spectra_tag,
            "notable_peak": notable_peak,
            "alias_tag": alias_tag,
            "chem_tag": chem_tag,
            "smiles": smiles,
            "formula": formula,
            "molecular_weight": molecular_weight,
            "ghs_codes": ghs_codes,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "rotatable_bonds": rotatable_bonds,
            "metadata_block": metadata_block,
        }


    def _extract_peak(self, spectra_raw: Dict[str, Any]) -> str:
        """Devuelve un pico representativo (λmax o m/z) de los datos espectrales."""
        nm_peaks: List[float] = []
        mz_peaks: List[float] = []

        for section in spectra_raw.values():
            if not isinstance(section, list):
                continue
            for item in section:
                value = item.get("Value", {}) if isinstance(item, dict) else {}
                lines = value.get("StringWithMarkup") or []
                for seg in lines:
                    text = seg.get("String", "")
                    # Captura *todos* los números con unidad nm o m/z
                    nm_peaks.extend(
                        float(v) for v in re.findall(r"(\d+(?:\.\d+)?)\s*nm", text, flags=re.I)
                    )
                    mz_peaks.extend(
                        float(v) for v in re.findall(r"(\d+(?:\.\d+)?)\s*m\/?z", text, flags=re.I)
                    )

        if nm_peaks:
            nm_peaks.sort()                   # orden ascendente
            # Elige el primer λ ≥ 260 nm; si no hay, el mayor disponible
            chosen_nm = next((v for v in nm_peaks if v >= 260), nm_peaks[-1])
            return f"{chosen_nm:.0f} nm"

        if mz_peaks:
            return f"m/z {max(mz_peaks):.0f}"

        return ""

