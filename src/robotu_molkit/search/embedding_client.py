import logging
from typing import Optional, List
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import Credentials

class WatsonxEmbeddingClient:
    """
    Wrap IBM watsonx-ai SDK Embeddings model to embed free-text queries.
    """
    def __init__(
        self,
        api_key: str,
        project_id: str,
        ibm_url: str,
        model: str,
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.ibm_url = ibm_url.rstrip('/')
        self.embedder = Embeddings(
            model_id=model,
            credentials=Credentials(api_key=self.api_key, url=self.ibm_url),
            project_id=self.project_id,
        )

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Return the embedding vector for a single text string.
        """
        try:
            # Watsonx expects a list of strings; take the first (and only) vector.
            return self.embedder.embed_documents(texts=[text],)[0]
        except Exception as exc:
            logging.warning("Embedding error: %s", exc)
            return None
