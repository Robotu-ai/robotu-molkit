import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from robotu_molkit.constants import DEFAULT_EMBED_MODEL_ID, DEFAULT_WATSONX_AI_URL

class WatsonxIndex:
    """
    Class to manage embedding ingestion and search operations in Watsonx Vector DB.
    """
    def __init__(
        self,
        api_key: str,
        project_id: str,
        ibm_url: str = DEFAULT_WATSONX_AI_URL,
        model: str = DEFAULT_EMBED_MODEL_ID,
        chunk_size: int = 250,
        overlap: int = 40,
    ):
        # Credentials and embedding service
        self.credentials = Credentials(api_key=api_key, url=ibm_url)
        self.embed_service = Embeddings(
            model_id=model,
            credentials=self.credentials,
            project_id=project_id,
        )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_cids(
        self,
        cids: List[int],
        parsed_dir: Path,
    ) -> None:
        """
        For each CID:
          1. Load the parsed JSON (parsed_dir/{cid}.json).
          2. Generate a global summary and its embedding.
          3. Split each thematic section into chunks and generate embeddings.
          4. (TODO) Upload each vector with metadata to Watsonx Vector DB.
        """
        for cid in cids:
            path = Path(parsed_dir) / f"pubchem_{cid}.json"
            if not path.exists():
                raise FileNotFoundError(f"File not found for CID {cid}: {path}")
            data = json.loads(path.read_text())

            # 1) Global summary
            summary_text = self._generate_summary(data)
            summary_emb = self._get_embeddings([summary_text])[0]
            # TODO: upload summary_emb with metadata {"cid": cid, "section": "summary"}

            # 2) Thematic sections
            for section, text in self._iter_sections(data):
                chunks = self._chunk_text(text)
                embeddings = self._get_embeddings(chunks)
                for chunk, emb in zip(chunks, embeddings):
                    # TODO: upload emb with metadata {"cid": cid, "section": section}
                    pass

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes an embedding search:
          - Obtains the embedding for the query.
          - Calls Watsonx Vector DB with any provided filters.
          - Returns a list of hits grouped by CID.
        """
        q_emb = self._get_embeddings([query])[0]
        # TODO: implement real call to the vector index:
        # response = self.vector_db.query(...)
        return []

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calls the Watsonx embedding service and returns the list of vectors.
        """
        vector = None
        try:
            vector = self.embed_service.embed_documents(texts=texts)[0]
            print("Vector:", vector)
        except Exception as e:
            logging.warning("Embedding error: %s", e)
        return vector

    def _iter_sections(self, data: Dict[str, Any]):
        """
        Generates (section, text) pairs for the molecule categories.
        """
        sections = ["structure", "safety", "spectra", "solubility", "search"]
        for sec in sections:
            text = json.dumps(data.get(sec, {}))
            yield sec, text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Splits the text into chunks of `chunk_size` tokens with `overlap`.
        """
        tokens = text.split()
        step = self.chunk_size - self.overlap
        chunks: List[str] = []
        for i in range(0, len(tokens), step):
            chunk = " ".join(tokens[i : i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(tokens):
                break
        return chunks

    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """
        Generates a global summary of the molecule (40–70 words).
        Implement your own logic here or call a generative model.
        """
        name = data.get("names", {}).get("preferred", "Unknown")
        cid = data.get("search", {}).get("cid")
        return f"Molecule {name} (CID {cid}): summary placeholder."

