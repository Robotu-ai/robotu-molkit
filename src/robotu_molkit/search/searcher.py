import json
import re
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from robotu_molkit.credentials_manager import CredentialsManager
from robotu_molkit.search.embedding_client import WatsonxEmbeddingClient
from robotu_molkit.search.index_manager import FAISSIndexManager
from robotu_molkit.constants import (
    DEFAULT_WATSONX_AI_URL,
    DEFAULT_EMBED_MODEL_ID,
    DEFAULT_WATSONX_GENERATIVE_MODEL
)

from pubchempy import get_compounds
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

class QueryRefiner:
    """Utilities for scaffold inference and result post-processing."""

    @staticmethod
    def extract_scaffolds_with_granite(model: ModelInference, query_text: str) -> List[str]:
        prompt = (
            "You are a molecular search assistant.\n"
            "Extract only the canonical names of up to three well-known molecules that structurally represent the query, "
            f"using common trivial names. Query: '{query_text}'.\n"
            "Return only this JSON exactly with no extra text:\n"
            "{\n  \"canonical_names\": [\"...\"]\n}"
        )
        response = model.generate_text(prompt=prompt)
        # attempt to parse JSON at start
        try:
            obj, _ = json.JSONDecoder().raw_decode(response)
            names = obj.get("canonical_names", [])
            if isinstance(names, list):
                return [n.strip() for n in names if isinstance(n, str) and n.strip()]
        except Exception:
            pass
        # fallback regex
        match = re.search(r"\{[^{}]*?\"canonical_names\"[^{}]*?\}", response, flags=re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
                names = obj.get("canonical_names", [])
                if isinstance(names, list):
                    return [n.strip() for n in names if isinstance(n, str) and n.strip()]
            except Exception:
                pass
        return []

    @staticmethod
    def resolve_scaffolds_to_bitvectors(scaffold_names: List[str], searcher: 'LocalSearch') -> List[np.ndarray]:
        """
        Resolve scaffold names via PubChem, fetch metadata, and build ECFP numpy bit arrays.
        """
        vectors: List[np.ndarray] = []
        for name in scaffold_names:
            try:
                compounds = get_compounds(name, "name", listkey_count=1)
                if not compounds:
                    continue
                cid = compounds[0].cid
                meta = searcher.get(cid)
                arr = np.zeros(1024, dtype=int)
                for i in meta.get("ecfp", []):
                    if 0 <= i < 1024:
                        arr[i] = 1
                vectors.append(arr)
                print(f"✅ CID {cid} for '{name}' → ECFP vector loaded")
            except Exception as e:
                print(f"⚠️ Failed to resolve scaffold '{name}': {e}")
        return vectors

    # Utility functions for ECFP and Tanimoto

    def ecfp_bits_from_meta(meta: Dict[str, Any]) -> np.ndarray:
        """Convert stored ECFP indices into a 1024-bit numpy array."""
        arr = np.zeros(1024, dtype=int)
        for i in meta.get("ecfp", []):
            if 0 <= i < 1024:
                arr[i] = 1
        return arr

    def tanimoto_bits(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Tanimoto similarity between two binary numpy arrays."""
        common = np.bitwise_and(a, b).sum()
        total  = np.bitwise_or(a, b).sum()
        return common / total if total else 0.0

class LocalSearch:
    """Local semantic search with metadata filters and structural refinement via Tanimoto."""
    def __init__(
        self,
        jsonl_path: str,
        override_api_key: Optional[str] = None,
        override_project_id: Optional[str] = None,
        ibm_url: str = DEFAULT_WATSONX_AI_URL,
        embed_model_id: str = DEFAULT_EMBED_MODEL_ID
    ):
        api_key, project_id = CredentialsManager.load(override_api_key, override_project_id)
        if not api_key or not project_id:
            raise ValueError("IBM Watsonx credentials not provided.")
        self.api_key = api_key
        self.project_id = project_id
        self.ibm_url = ibm_url
        self.embed_client = WatsonxEmbeddingClient(api_key=api_key, project_id=project_id,
                                                   ibm_url=ibm_url, model=embed_model_id)
        path = Path(jsonl_path)
        first = json.loads(path.open().read().splitlines()[0])
        dim = len(first.get("vector", []))
        self.index = FAISSIndexManager(dim)
        self.index.load_jsonl(path)

    def get(self, cid: int) -> Dict[str, Any]:
        for meta in self.index.metadata:
            if meta.get("cid") == cid:
                return meta
        raise KeyError(f"CID {cid} not found in index.")

    def query(self, text: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None,
              faiss_k: int = 100) -> List[Tuple[Dict[str, Any], float]]:
        qvec = self.embed_client.embed(text)
        if qvec is None:
            return []
        qarr = np.array(qvec, dtype="float32")
        hits = self.index.search(qarr, faiss_k)
        if not filters:
            return hits[:top_k]
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
        return [(m, s) for m, s in hits if passes(m)][:top_k]

    def query_with_tanimoto(self, query_text: str, top_k: int = 20,
                            faiss_k: int = 300, filters: Optional[Dict[str, Any]] = None,
                            sim_threshold: float = 0.7) -> List[Tuple[Dict[str, Any], float, float]]:
        # Step 1: Setup Granite
        creds = Credentials(api_key=self.api_key, url=self.ibm_url)
        model = ModelInference(model_id=DEFAULT_WATSONX_GENERATIVE_MODEL, credentials=creds,
                               project_id=self.project_id, params={GenParams.MAX_NEW_TOKENS:500, GenParams.TEMPERATURE:0.2})
        # Step 2: Infer scaffold names
        scaffold_names = QueryRefiner.extract_scaffolds_with_granite(model, query_text)
        print("🔍 Inferred scaffolds:", scaffold_names)
        # Step 3: Build reference ECFP vectors
        ref_vecs = QueryRefiner.resolve_scaffolds_to_bitvectors(scaffold_names, self)
        # Step 4: Raw semantic search
        raw = self.query(text=query_text, top_k=faiss_k, filters=filters, faiss_k=faiss_k)
        # Step 5: Filter by Tanimoto
        results: List[Tuple[Dict[str, Any], float, float]] = []
        for meta, score in raw:
            mol_vec = ecfp_bits_from_meta(meta)
            sims = [tanimoto_bits(mol_vec, ref) for ref in ref_vecs]
            max_sim = max(sims) if sims else 0.0
            print(f"→ CID {meta.get('cid')} Tanimoto: {max_sim:.2f}")
            if max_sim >= sim_threshold:
                results.append((meta, score, max_sim))
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"✅ {len(results)} of {len(raw)} passed Tanimoto ≥ {sim_threshold}")
        return results[:top_k]




