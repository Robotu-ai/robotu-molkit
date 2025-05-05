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
API_KEY    = ""
PROJECT_ID = ""

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
        "methylxanthine derivatives with central nervous system stimulant activity"
    )
    filters = {
        "molecular_weight": (0, 250),
        "solubility_tag": "soluble"
    }

    # Perform semantic search only
    results1 = searcher.search_by_semantics(
        query_text=query_text,
        top_k=TOP_K,
        faiss_k=FAISS_K,
        filters=filters,
    )

    print(f"\nTop {len(results1)} hits from semantic-only search:")
    print(f"▼ (Unfiltered by structure)\n")
    for meta, score in results1:
        print(
            f"CID {meta['cid']:<6} "
            f"Name: {meta.get('name','<unknown>'):<25} "
            f"MW: {meta.get('molecular_weight',0):<6.1f} "
            f"Solubility: {meta.get('solubility_tag',''):<10} "
            f"Score: {score:.3f}"
        )

    # Perform structural+semantic search with Tanimoto refinement
    results = searcher.search_by_semantics_and_structure(
        query_text=query_text,
        top_k=TOP_K,
        faiss_k=FAISS_K,
        filters=filters,
        sim_threshold=SIM_THRESHOLD
    )

    print(f"\nTop {len(results)} hits after semantic+structural filtering:")
    print(f"▼ (Granite scaffolds, Tanimoto ≥ {SIM_THRESHOLD})\n")
    for meta, score, sim in results:
        print(
            f"CID {meta['cid']:<6} "
            f"Name: {meta.get('name','<unknown>'):<25} "
            f"MW: {meta.get('molecular_weight',0):<6.1f} "
            f"Solubility: {meta.get('solubility_tag',''):<10} "
            f"Score: {score:.3f} "
            f"Tanimoto: {sim:.2f}"
        )

if __name__ == "__main__":
    main()



