"""
Example: Use LocalSearch.search_by_semantics_and_structure to find molecules related to a user query.
Granite Instruct infers scaffolds and Tanimoto filtering is applied automatically.
"""
from robotu_molkit.credentials_manager import CredentialsManager
from robotu_molkit.search.searcher import LocalSearch
from robotu_molkit.constants import DEFAULT_JSONL_FILE_ROUTE

# --------------------------------------------------------------------------- #
# Configuration WATSONX CREDENTIALS                                          #
# --------------------------------------------------------------------------- #
API_KEY    = "WBLC6RY7mWwVGdVqWY2GVPfJ_yQjM_HzQJH5GVlFoXUp"
PROJECT_ID = "13d1284a-8dae-4a88-b776-3b890b249f2d"

# Persist credentials for all LocalSearch calls
CredentialsManager.set_api_key(API_KEY)
CredentialsManager.set_project_id(PROJECT_ID)

JSONL_PATH    = DEFAULT_JSONL_FILE_ROUTE
SIM_THRESHOLD = 0.70
TOP_K         = 20
FAISS_K       = 300
# --------------------------------------------------------------------------- #

def main():
    # Initialize searcher
    searcher = LocalSearch(jsonl_path=JSONL_PATH)

    # Define query and metadata filters
    query_text = (
        "Methylxanthine derivatives with central nervous system stimulant activity"
    )
    filters = {
        "molecular_weight": (0, 250),
        "solubility_tag": "soluble"
    }

    # Perform semantic + structural search
    results = searcher.search_by_semantics_and_structure(
        query_text=query_text, top_k=20, faiss_k=300, filters=filters, sim_threshold=0.70
    )

    # Format and display results
    entries = [
        f"CID {m['cid']} Name:{m.get('name','<unknown>')} MW:{m.get('molecular_weight',0):.1f} "
        f"Sol:{m.get('solubility_tag','')} Score:{s:.3f} Tanimoto:{sim:.2f}"
        for m, s, sim in results
    ]

    print(
        f"Results for query: \"{query_text}\"\n"
        f"Top {len(entries)} hits (Granite-inferred scaffolds, Tanimoto ≥ {SIM_THRESHOLD}):\n"
        + "\n".join(entries)
        + "\n\nNote: Scaffold inference was performed using IBM's granite-3-8b-instruct model. "
        "Semantic and structural similarity search was powered by granite-embedding-278m-multilingual."
    )


if __name__ == "__main__":
    main()



