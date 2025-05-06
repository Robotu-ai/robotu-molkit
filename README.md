# 🧪 robotu-molkit

> 🚧 **This project is under active development.** Expect frequent changes as we build the foundation for quantum-ready molecular discovery.

**Quantum-ready molecular toolkit.**  
robotu-molkit is the first library to enrich PubChem molecules with AI-native context: each molecule is converted into a simulation-ready `Molecule` object and annotated using IBM Granite models to generate both human-readable summaries and high-dimensional embeddings—bridging chemistry, AI, and quantum workflows.

---

## 🔍 About

**robotu-molkit** is part of the **RobotU Quantum** ecosystem, and it's the first open-source toolkit to unify molecular data curation, AI enrichment, and semantic search—designed from the ground up for quantum and AI workflows.

Unlike traditional cheminformatics libraries, robotu-molkit goes beyond parsing: it integrates **IBM watsonx Granite models** to generate natural-language summaries and high-dimensional vector embeddings for each molecule. These AI-generated fingerprints capture not just structure, but meaning—enabling search queries like *"low-toxicity CNS stimulants under 250 Da"* to return relevant results instantly.

robotu-molkit ingests records from **PubChem**, standardizes >10 property categories (geometry, quantum, spectra, safety, solubility, etc.), and outputs clean `Molecule` objects with embedded context-aware vectors. Molecules can be searched semantically, compared structurally, or exported into simulation pipelines—making it ideal for researchers in quantum chemistry, drug discovery, and AI-accelerated science.

It’s the first library to:
- Embed both summaries and molecular sections using **Granite Embedding**
- Enable similarity search powered by **local FAISS** (Milvus vector database throug watsonx, comming soon).
- Support hybrid semantic + structure-based filtering via **Tanimoto + AI vectors**

In short: robotu-molkit turns raw chemical records into **simulation-ready, AI-searchable molecules**.

## 📦 Installation

```bash
pip install robotu-molkit
```

---

## 🛠️ CLI Usage

robotu‑molkit ships with a single entry‑point, **`molkit`**, that orchestrates each pipeline stage.

> ℹ️ Run `molkit --help` or `molkit <command> --help` for full option details.

### 0. Configure (one‑time)

```bash
molkit config --watsonx-api-key $WATSONX_API_KEY --watsonx-project-id $WATSONX_PROJECT_ID
```

### 1. Ingest — download & parse PubChem records

```bash
molkit ingest 2244 1983 3675
molkit ingest --file path/to/cids.txt
molkit ingest 2244 1983 --concurrency 8
```

### 2. Embed — enrich with Granite summaries & vectors

```bash
molkit embed
molkit embed --fast
```

### 3. Upload — *not yet implemented*

Currently stops after generating `watsonx_vectors.jsonl`.

---

## 🧬 Available Fields

### 🧾 Identifiers and Names
- `name`, `inchi`, `inchikey`, `smiles`, `cid`, `formula`, `molecular_weight`

### ⚛️ Structure and Geometry
- `xyz`, `heavy_atom_count`, `ring_count`, `aromatic_ring_count`, `rotatable_bonds`, `fsp3`, `bertz_ct`

### 🧪 Properties
- `hbond_donors`, `hbond_acceptors`, `tpsa`, `logp`, `logs`, `ghs_codes`, `hazard_tag`, `solubility_tag`, `spectra_tag`, `chem_tag`

### 🧠 Embeddings and Metadata
- `summary`, `structure`, `ecfp`, `maccs`

**Possible `solubility_tag` values and their thresholds:**

- `unknown solubility`  
  - When log‐solubility (`logs`) is `None`.

- `very soluble`  
  - `logs > -0.5`

- `soluble`  
  - `-1.5 < logs ≤ -0.5`

- `moderately soluble`  
  - `-3.0 < logs ≤ -1.5`

- `sparingly soluble`  
  - `-4.0 < logs ≤ -3.0`

- `insoluble`  
  - `logs ≤ -4.0`

---

## 💡 Search Examples

```python
from robotu_molkit.credentials_manager import CredentialsManager
from robotu_molkit.search.searcher import LocalSearch
from robotu_molkit.constants import DEFAULT_JSONL_FILE_ROUTE

WATSON_API_KEY = ""
WATSON_PROJECT_ID = ""
CredentialsManager.set_api_key(WATSON_API_KEY)
CredentialsManager.set_project_id(WATSON_PROJECT_ID)

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

```

## Parameter `filters`

The `filters` parameter of `search_by_semantics` and `search_by_semantics_and_structure` allows you to refine results based on metadata. It’s a Python `dict` mapping field names to conditions:

`python
filters: Dict[str, Any] = {
    'field': condition,
    # …
}
`

### Condition types

- **Single value** (equality)  
  `python
  filters = { 'solubility_tag': 'High' }
  `  
  Only entries where `meta['solubility_tag'] == 'High'` pass.

- **Range** (`tuple`)  
  `python
  filters = { 'molecular_weight': (100, 500) }
  `  
  Only entries where `100 <= meta['molecular_weight'] <= 500` pass.

- **List** (membership)  
  `python
  filters = { 'cid': [119, 971, 1123] }
  `  
  Only entries where `meta['cid']` is in the list pass.

---

Internally, filtering is done like this:

```python
def passes(m: Dict[str, Any]) -> bool:
    for k, cond in filters.items():
        v = m.get(k)
        if isinstance(cond, tuple):
            if v is None or not (cond[0] <= v <= cond[1]):
                return False
        elif isinstance(cond, list):
            if v not in cond:
                return False
        else:
            if v != cond:
                return False
    return True

filtered = [(m, s) for m, s in hits if passes(m)][:top_k]
```

### Example usage

```python
my_filters = {
    'solubility_tag': 'soluble',
    'molecular_weight': (100, 250),
}

results = searcher.search_by_semantics(
    query_text="molecules structurally or functionally similar to caffeine",
    top_k=20,
    filters=my_filters
)
```


---

## 📄 License

Apache 2.0 License — see LICENSE file.

---

**RobotU Quantum — accelerating discovery through open, AI-enhanced, quantum-ready data.**



