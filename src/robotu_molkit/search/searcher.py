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

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

class QueryRefiner:
    """Utilities for scaffold inference and result post-processing."""

    @staticmethod
    def extract_scaffolds_with_granite(model: ModelInference, query_text: str) -> List[str]:
        prompt = (
            "You are an expert molecular search assistant.\n"
            "Identify and return only the common/trivial English names of up to three well-known molecules "
            "that structurally represent the following query, using their most widely recognized names (e.g., caffeine, theophylline, theobromine):\n"
            f"\"{query_text}\"\n"
            "Provide the output strictly in this JSON format without any extra text or numbering:\n"
            "{\n  \"canonical_names\": [\"caffeine\", \"theophylline\", \"theobromine\"]\n}"
        )
        response = model.generate_text(prompt=prompt)
        print("response granite: ", response)
        return QueryRefiner._extract_json_list(response, "canonical_names")

    @staticmethod
    def resolve_scaffolds_to_bitvectors(scaffold_names: List[str], searcher: 'LocalSearch') -> List[ExplicitBitVect]:
        """
        Resolve scaffold names to PubChem CIDs, fetch metadata, and return RDKit ExplicitBitVect fingerprints.
        """
        vectors: List[ExplicitBitVect] = []
        for name in scaffold_names:
            try:
                compounds = get_compounds(name, "name", listkey_count=1)
                if not compounds:
                    continue
                cid = compounds[0].cid
                meta = searcher.get(cid)
                # Build bit vector from stored indices
                bv = ExplicitBitVect(1024)
                for i in meta.get("ecfp", []):
                    if 0 <= i < 1024:
                        bv.SetBit(i)
                vectors.append(bv)
                print(f"✅ CID {cid} for '{name}' → scaffold fingerprint loaded")
            except Exception as e:
                print(f"⚠️  Failed to resolve scaffold '{name}': {e}")
        return vectors

    @staticmethod
    def _extract_json_list(txt: str, key: str) -> List[str]:
        """Extract list from the first JSON object containing the given key, even if extra text follows."""
        # Try raw JSON decode
        try:
            obj, _ = json.JSONDecoder().raw_decode(txt)
            vals = obj.get(key)
            if isinstance(vals, list):
                return [s.strip() for s in vals if isinstance(s, str) and s.strip()]
        except Exception:
            pass
        # Fallback regex
        match = re.search(r"\{[^{}]*?\"" + re.escape(key) + r"\"[^{}]*?\}", txt, flags=re.S)
        if match:
            block = match.group(0)
            try:
                obj = json.loads(block)
                vals = obj.get(key, [])
                if isinstance(vals, list):
                    return [s.strip() for s in vals if isinstance(s, str) and s.strip()]
            except Exception:
                pass
        return []

class LocalSearch:
    """
    Local semantic search over molecular embeddings with metadata filters and structural similarity (Tanimoto).
    """
    def __init__(
        self,
        jsonl_path: str,
        override_api_key: Optional[str] = None,
        override_project_id: Optional[str] = None,
        ibm_url: str = DEFAULT_WATSONX_AI_URL,
        embed_model_id: str = DEFAULT_EMBED_MODEL_ID
    ):
        api_key, project_id = CredentialsManager.load(
            override_api_key, override_project_id
        )
        if not api_key or not project_id:
            raise ValueError("IBM Watsonx credentials not provided.")
        self.api_key = api_key
        self.project_id = project_id
        self.ibm_url = ibm_url
        self.embed_client = WatsonxEmbeddingClient(
            api_key=api_key,
            project_id=project_id,
            ibm_url=ibm_url,
            model=embed_model_id
        )
        path = Path(jsonl_path)
        first = json.loads(path.open().read().splitlines()[0])
        dim = len(first["vector"])
        self.index = FAISSIndexManager(dim)
        self.index.load_jsonl(path)

    def get(self, cid: int) -> Dict[str, Any]:
        for meta in self.index.metadata:
            if meta.get("cid") == cid:
                return meta
        raise KeyError(f"CID {cid} not found in index.")

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        faiss_k: int = 100
    ) -> List[Tuple[Dict[str, Any], float]]:
        qvec = self.embed_client.embed(query_text)
        if qvec is None:
            return []
        qarr = np.array(qvec, dtype="float32")
        hits = self.index.search(qarr, faiss_k)
        if not filters:
            return hits[:top_k]
        def passes(meta: Dict[str, Any]) -> bool:
            for key, cond in filters.items():
                val = meta.get(key)
                if isinstance(cond, tuple):
                    if val is None or not (cond[0] <= val <= cond[1]):
                        return False
                elif isinstance(cond, list):
                    if val not in cond:
                        return False
                else:
                    if val != cond:
                        return False
            return True
        filtered = [(m, s) for m, s in hits if passes(m)]
        return filtered[:top_k]

    def query_with_tanimoto(
        self,
        query_text: str,
        top_k: int = 20,
        faiss_k: int = 300,
        filters: Optional[Dict[str, Any]] = None,
        sim_threshold: float = 0.7
    ) -> List[Tuple[Dict[str, Any], float, float]]:
        # Setup Granite model
        creds = Credentials(api_key=self.api_key, url=self.ibm_url)
        model = ModelInference(
            model_id=DEFAULT_WATSONX_GENERATIVE_MODEL,
            credentials=creds,
            project_id=self.project_id,
            params={GenParams.MAX_NEW_TOKENS: 500, GenParams.TEMPERATURE: 0.2}
        )
        # Infer scaffolds
        scaffold_names = QueryRefiner.extract_scaffolds_with_granite(model, query_text)
        print("🔍 Inferred scaffolds:", scaffold_names)
        # Convert scaffolds to RDKit bit vectors
        ref_bvs = QueryRefiner.resolve_scaffolds_to_bitvectors(scaffold_names, self)
        # Perform semantic query
        raw_results = self.query(
            query_text=query_text,
            top_k=faiss_k,
            filters=filters,
            faiss_k=faiss_k
        )
        # Filter by structural similarity
        results: List[Tuple[Dict[str, Any], float, float]] = []
        for meta, score in raw_results:
            # Build explicit bit vect for candidate
            bv_candidate = ExplicitBitVect(1024)
            for i in meta.get("ecfp", []):
                if 0 <= i < 1024:
                    bv_candidate.SetBit(i)
            # Compute max Tanimoto vs scaffolds
            sims = [DataStructs.TanimotoSimilarity(bv_candidate, bv_ref) for bv_ref in ref_bvs]
            max_sim = max(sims) if sims else 0.0
            print(f"→ CID {meta.get('cid')} Tanimoto max: {max_sim:.2f}")
            if max_sim >= sim_threshold:
                results.append((meta, score, max_sim))
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"✅ {len(results)} of {len(raw_results)} passed Tanimoto ≥ {sim_threshold}")
        return results[:top_k]



