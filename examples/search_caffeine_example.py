"""
Example: Use LocalSearch to find molecules related to a user query.
Granite Instruct is used to interpret the query and infer one or more scaffolds.
Then, Tanimoto filtering is applied against scaffold(s) retrieved from PubChem.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Set, List, Sequence

import numpy as np
import json
import re
from pubchempy import get_compounds
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from robotu_molkit.credentials_manager import CredentialsManager
from robotu_molkit.search.searcher import LocalSearch
from robotu_molkit.constants import (
    DEFAULT_JSONL_FILE_ROUTE,
    DEFAULT_WATSONX_AI_URL,
    DEFAULT_WATSONX_GENERATIVE_MODEL
)

# --------------------------------------------------------------------------- #
# Configuration                                                              #
# --------------------------------------------------------------------------- #
API_KEY = "WBLC6RY7mWwVGdVqWY2GVPfJ_yQjM_HzQJH5GVlFoXUp"
PROJECT_ID = "13d1284a-8dae-4a88-b776-3b890b249f2d"

CredentialsManager.set_api_key(API_KEY)
CredentialsManager.set_project_id(PROJECT_ID)
CredentialsManager.set_watsonx_url(DEFAULT_WATSONX_AI_URL)

JSONL_PATH = DEFAULT_JSONL_FILE_ROUTE
SIM_THRESHOLD = 0.70
TOP_K = 20
FAISS_K = 300
GRANITE_MODEL_ID = DEFAULT_WATSONX_GENERATIVE_MODEL
# --------------------------------------------------------------------------- #

def _extract_json_list(txt: str, key: str) -> List[str]:
    """Extract list from the first JSON object containing the given key."""
    json_match = re.search(r"\{[^{}]*?\"" + re.escape(key) + r"\"[^{}]*?\}", txt, flags=re.S)
    if json_match:
        block = json_match.group(0)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
                return [s.strip() for s in obj[key] if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            try:
                obj = json.loads(block.replace("'", '"'))
                return [s.strip() for s in obj.get(key, []) if isinstance(s, str) and s.strip()]
            except Exception:
                pass
    return []

def tanimoto_bits(a: np.ndarray, b: np.ndarray) -> float:
    common = np.bitwise_and(a, b).sum()
    total = np.bitwise_or(a, b).sum()
    return common / total if total else 0.0

def ecfp_bits_from_meta(meta: Dict) -> np.ndarray:
    return np.array(meta["ecfp"], dtype=int)  # ← fix dtype to avoid uint8 overflow

def extract_scaffold_names_via_granite(model: ModelInference, query: str) -> List[str]:
    prompt = (
        f"You are a molecular search assistant."
        f"Extract only the canonical names of up to three well-known molecules that structurally represent the query:"
        f"\"{query}\""
        f"Return only this response Format:{{\"canonical_names\": [\"...\"]}}"
    )
    response = model.generate_text(prompt=prompt)
    return _extract_json_list(response, "canonical_names")

def main():
    # ------------------------------------------------------------------------
    creds = Credentials(api_key=API_KEY, url=DEFAULT_WATSONX_AI_URL)
    model = ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=creds,
        project_id=PROJECT_ID,
        params={GenParams.MAX_NEW_TOKENS: 500, GenParams.TEMPERATURE: 0.2}
    )

    searcher = LocalSearch(jsonl_path=JSONL_PATH)

    # ------------------------------------------------------------------------
    full_query = (
        "methylxanthine derivatives with central nervous system stimulant activity"
    )
    filters = {
        "molecular_weight": (0, 250),
        "solubility_tag": "soluble"
    }

    # ------------------------------------------------------------------------
    scaffold_names = extract_scaffold_names_via_granite(model, full_query)
    print("🔍 Inferred scaffolds:", scaffold_names)

    ref_bits_list = []
    for name in scaffold_names:
        try:
            compound = get_compounds(name, "name", listkey_count=1)
            if compound:
                cid = compound[0].cid
                meta = searcher.get_metadata(cid=cid)
                bits = ecfp_bits_from_meta(meta)
                ref_bits_list.append(bits)
        except Exception:
            continue

    # ------------------------------------------------------------------------
    raw_results = searcher.query(
        text=full_query,
        top_k=FAISS_K,
        filters=filters,
        faiss_k=FAISS_K
    )

    filtered_results: List[Tuple[Dict[str, Any], float, float]] = []
    for meta, score in raw_results:
        mol_bits = ecfp_bits_from_meta(meta)
        if ref_bits_list:
            if any(tanimoto_bits(mol_bits, ref) >= SIM_THRESHOLD for ref in ref_bits_list):
                sim = max(tanimoto_bits(mol_bits, ref) for ref in ref_bits_list)
                filtered_results.append((meta, score, sim))
        if not ref_bits_list:
            print("⚠️  No valid scaffolds retrieved. Skipping Tanimoto filtering.")
            return

    filtered_results.sort(key=lambda x: x[1], reverse=True)
    results = filtered_results[:TOP_K]

    print(f"\nTop {len(results)} hits for query:\n  '{full_query}'")
    print(f"▼ (Granite inferred scaffolds, Tanimoto ≥ {SIM_THRESHOLD})\n")
    for meta, score, sim in results:
        cid = meta.get("cid")
        name = meta.get("name", '<unknown>')
        smiles = meta.get("smiles", '<no smiles>')
        mw = meta.get("molecular_weight")
        sol = meta.get("solubility_tag")
        print(
            f"CID {cid:<6} Name: {name:<25} MW: {mw:<7.1f} "
            f"Solubility: {sol:<14} Score: {score:.3f} Tanimoto: {sim:.2f}"
        )

if __name__ == "__main__":
    main()

