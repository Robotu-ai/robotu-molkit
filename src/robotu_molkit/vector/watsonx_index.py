
# watsonx_index.py  –  streamlined, folder‑based ingestor
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from robotu_molkit.constants import DEFAULT_EMBED_MODEL_ID, DEFAULT_WATSONX_AI_URL
from robotu_molkit.vector.summary_generator import SummaryGenerator

CID_RE = re.compile(r"(\d+)")          # captures digits in “pubchem_2519.json”


class WatsonxIndex:
    """
    Walk through a folder with parsed PubChem JSON files and create a
    JSONL file containing: {"cid": …, "vector": […], "text": …} for each molecule.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        ibm_url: str = DEFAULT_WATSONX_AI_URL,
        model: str = DEFAULT_EMBED_MODEL_ID,
    ) -> None:
        # Watsonx embedding client
        self.embedder = Embeddings(
            model_id=model,
            credentials=Credentials(api_key=api_key, url=ibm_url),
            project_id=project_id,
        )
        # Generates the single “general” blurb; credentials auto‑loaded
        self.sg = SummaryGenerator()

    # ------------------------------------------------------------------ #
    # Folder ingestion                                                   #
    # ------------------------------------------------------------------ #
    def ingest_folder(
        self,
        parsed_dir: Path,
        out_dir: Path = Path("data/vectors"),
        pattern: str = "pubchem_*.json",
    ) -> Path:
        """
        • Scan `parsed_dir` for JSON files matching *pattern*.
        • For each file:
            1. Build a general summary.
            2. Get its embedding vector.
            3. Extract filterable metadata from the same parsed JSON.
            4. Append one JSON line to <out_dir>/watsonx_vectors.jsonl
        Returns the path to the resulting JSONL file.
        """
        files = list(parsed_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No parsed JSON found in {parsed_dir} using pattern '{pattern}'"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "watsonx_vectors.jsonl"

        with jsonl_path.open("w", encoding="utf-8") as sink:
            for file_path in files:
                cid = self._cid_from_filename(file_path)
                if cid is None:
                    logging.warning("Skipping file without CID in name: %s", file_path.name)
                    continue

                data = json.loads(file_path.read_text())

                # 1) Summary
                summary = self.sg.generate_general_summary(data)
                if not summary:
                    logging.warning("Empty summary for CID %s – skipped", cid)
                    continue

                # 2) Embedding
                vector = self._embed(summary)
                if vector is None:
                    continue

                # 3) Metadata extraction
                names  = data.get("names", {})
                search  = data.get("search", {})
                sol     = data.get("solubility", {})
                safety  = data.get("safety", {})
                thermo  = data.get("thermo", {})
                meta    = data.get("meta", {})
                spectra = data.get("spectra", {}).get("raw", {}) or {}
                pka_vals = sol.get("pka", []) or []
                structure = data.get("structure", []) or []

                # 4) Expectra interpretation 
                spectra_tag, notable_peak = self.sg.format_spectra_info(spectra)

                record: Dict[str, Any] = {
                    # identifiers & summary/vector
                    "cid":      cid,
                    "summary":  summary,
                    "vector":   vector,
                    "name": names.get("preferred", None),

                    # search‐section fields
                    "inchi":                search.get("inchi"),
                    "inchikey":             search.get("inchikey"),
                    "smiles":               search.get("smiles"),
                    "molecular_weight":     search.get("molecular_weight"),
                    "formula":              search.get("formula"),
                    "heavy_atom_count":     search.get("heavy_atom_count"),
                    "hbond_donors":         search.get("hbond_donors"),
                    "hbond_acceptors":      search.get("hbond_acceptors"),
                    "rotatable_bonds":      search.get("rotatable_bonds"),
                    "ring_count":           search.get("ring_count"),
                    "aromatic_ring_count":  search.get("aromatic_ring_count"),
                    "tpsa":                 search.get("tpsa"),
                    "fsp3":                 search.get("fsp3"),
                    "bertz_ct":             search.get("bertz_ct"),

                    # fingerprint arrays
                    "ecfp":  search.get("ecfp", []),
                    "maccs": search.get("maccs", []),

                    # quantitative metadata
                    "logp":      sol.get("logp"),
                    "logS":      sol.get("logs"),

                    # qualitative tags
                    "hazard_tag":     self.sg._qualitative_hazard(safety.get("ghs_codes", [])),
                    "solubility_tag": self.sg._qualitative_sol(sol.get("logs")),
                    "spectra_tag":    data.get("spectra", {}).get("raw", {}) and 
                                        ", ".join(k.replace(" Spectra", "") for k in data["spectra"]["raw"].keys()) 
                                        + " spectra available" 
                                    or "no spectra available",
                    "chem_tag":    meta.get("chem_tag", []),
                    "ghs_codes":   safety.get("ghs_codes", []),
                    "xyz": structure.get("xyz", []),
                }
                sink.write(json.dumps(record) + "\n")

        logging.info("Vector JSONL created → %s  (%d molecules)", jsonl_path, len(files))
        return jsonl_path


    # ------------------------------------------------------------------ #
    # Search stub                                                        #
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Placeholder – integrate with Watsonx Vector DB later.
        """
        logging.info("Search('%s', top_k=%d) – not implemented yet.", query, top_k)
        return []

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cid_from_filename(path: Path) -> Optional[int]:
        """Extracts the CID from filenames like 'pubchem_2519.json'."""
        m = CID_RE.search(path.stem)
        return int(m.group(1)) if m else None

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Return the embedding vector for a single text string.
        """
        try:
            # Watsonx expects a list of strings; take the first (and only) vector.
            return self.embedder.embed_documents(texts=[text])[0]
        except Exception as exc:          # pylint: disable=broad-except
            logging.warning("Embedding error: %s", exc)
            return None
