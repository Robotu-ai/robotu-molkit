"""
Example: Use LocalSearch to find caffeine-related molecules.
"""
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from robotu_molkit.credentials_manager import CredentialsManager
from robotu_molkit.search.searcher import LocalSearch
from robotu_molkit.constants import (
    DEFAULT_JSONL_FILE_ROUTE,
    DEFAULT_WATSONX_AI_URL,
)

def main():
    # ----------------------------------------------------------------
    # 1) Configure Watsonx credentials and service URL (persisted)
    #    Replace the placeholders with your actual IBM Watsonx details.
    CredentialsManager.set_api_key("WBLC6RY7mWwVGdVqWY2GVPfJ_yQjM_HzQJH5GVlFoXUp")
    CredentialsManager.set_project_id("13d1284a-8dae-4a88-b776-3b890b249f2d")
    CredentialsManager.set_watsonx_url(DEFAULT_WATSONX_AI_URL)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 2) Initialize LocalSearch (uses config internally)
    searcher = LocalSearch(
        jsonl_path=DEFAULT_JSONL_FILE_ROUTE
    )
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 3) Define a more descriptive, caffeine-free query:
    #    methylxanthine derivatives with central nervous system stimulant activity, with MW < 250 Da and soluble.
    #    query_text = "methylxanthine derivatives with central nervous system
    #    stimulant activity"
    #    filters = {
    #        "molecular_weight": (0, 250),
    #        "solubility_tag": "soluble"
    #    }

    query_text = (
        "methylxanthine derivatives with central nervous system stimulant activity"
    )
    filters = {
        "molecular_weight": (0, 250),
        "solubility_tag": "soluble"
    }
    top_k = 20
    faiss_k = 100
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 4) Perform the query and capture results
    results = searcher.query(
        text=query_text,
        top_k=top_k,
        filters=filters,
        faiss_k=100
    )
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 5) Display enriched result metadata
    print(f"Top {len(results)} hits for query:\n  '{query_text}'\n")
    for meta, score in results:
        cid = meta.get("cid")
        name = meta.get("name", '<unknown>')
        smiles = meta.get("smiles", '<no smiles>')
        mw = meta.get("molecular_weight")
        sol = meta.get("solubility_tag")
        print(
            f"CID {cid:<6} "
            f"Name: {name:<25} "
            f"SMILES: {smiles:<20} "
            f"MW: {mw:<7} "
            f"Solubility: {sol:<10} "
            f"Score: {score:.3f}"
        )
    # ----------------------------------------------------------------

if __name__ == "__main__":
    main()
