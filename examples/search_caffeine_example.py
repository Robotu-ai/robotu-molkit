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
    # 1) Set IBM Watsonx credentials and URL in config
    # Replace the placeholders with your actual credentials
    CredentialsManager.set_api_key("WBLC6RY7mWwVGdVqWY2GVPfJ_yQjM_HzQJH5GVlFoXUp")
    CredentialsManager.set_project_id("13d1284a-8dae-4a88-b776-3b890b249f2d")
    CredentialsManager.set_watsonx_url(DEFAULT_WATSONX_AI_URL)

    # 2) Initialize the LocalSearch without passing credentials explicitly
    searcher = LocalSearch(
        jsonl_path=DEFAULT_JSONL_FILE_ROUTE
    )

    # 3) Perform a semantic query for caffeine analogues under 250 Da
    query_text = "caffeine analogues under 250 Da"
    filters = {"molecular_weight": (0, 250)}
    top_k = 5

    results = searcher.query(query_text, top_k=top_k, filters=filters)

    # 4) Display results
    print(f"Top {len(results)} results for '{query_text}':\n")
    for meta, score in results:
        cid = meta.get('cid')
        name = meta.get('names', {}).get('preferred_name', '<unknown>')
        mw = meta.get('molecular_weight')
        print(f"CID {cid} | {name} | MW={mw} | score={score:.3f}")


if __name__ == "__main__":
    main()
