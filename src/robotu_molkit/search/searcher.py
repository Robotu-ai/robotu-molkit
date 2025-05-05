import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from robotu_molkit.credentials_manager import CredentialsManager
from robotu_molkit.search.embedding_client import WatsonxEmbeddingClient
from robotu_molkit.search.index_manager import FAISSIndexManager
from robotu_molkit.constants import DEFAULT_WATSONX_AI_URL, DEFAULT_EMBED_MODEL_ID

class LocalSearch:
    """
    Local semantic search over molecular embeddings with metadata filters.

    Methods:
        - query(text, top_k, filters) -> List[(metadata, score)]
    """
    def __init__(
        self,
        jsonl_path: str,
        override_api_key: Optional[str] = None,
        override_project_id: Optional[str] = None,
        ibm_url: str = DEFAULT_WATSONX_AI_URL,
        embed_model_id: str = DEFAULT_EMBED_MODEL_ID
    ):
        # Load Watsonx credentials
        api_key, project_id = CredentialsManager.load(
            override_api_key, override_project_id
        )
        if not api_key or not project_id:
            raise ValueError("IBM Watsonx credentials not provided.")

        # Initialize embedding client
        self.embed_client = WatsonxEmbeddingClient(
            api_key=api_key,
            project_id=project_id,
            ibm_url=ibm_url,
            model=embed_model_id
        )

        # Determine vector dimension from JSONL
        path = Path(jsonl_path)
        first = json.loads(path.open().read().splitlines()[0])
        dim = len(first["vector"])

        # Build FAISS index
        self.index = FAISSIndexManager(dim)
        self.index.load_jsonl(path)

    def query(
        self,
        text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        faiss_k: int = 100
    ) -> List[Tuple[Dict[str, Any], float]]:
        # Embed the query text
        qvec = self.embed_client.embed(text)
        if qvec is None:
            return []
        qarr = np.array(qvec, dtype="float32")

        # Perform broader FAISS search
        hits = self.index.search(qarr, faiss_k)

        # Apply metadata filters if any
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
