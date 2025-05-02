# src/robotu_molkit/vector/WatsonxIndex.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings

class WatsonxIndex:
    """
    Clase para gestionar ingest de embeddings y búsquedas en Watsonx Vector DB.
    """
    def __init__(
        self,
        api_key: str,
        project_id: str,
        ibm_url: str = "https://us-south.ml.cloud.ibm.com",
        model: str = "granite-embedding-278m-multilingual",
        chunk_size: int = 250,
        overlap: int = 40,
    ):
        # Credenciales y servicio de embeddings
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
        Para cada CID:
          1. Carga el JSON parseado (parsed_dir/{cid}.json).
          2. Genera un summary y su embedding.
          3. Trocea cada sección temática y genera embeddings.
          4. (TODO) Sube cada vector con metadatos a Watsonx Vector DB.
        """
        for cid in cids:
            path = Path(parsed_dir) / f"{cid}.json"
            if not path.exists():
                raise FileNotFoundError(f"No se encontró el archivo para CID {cid}: {path}")
            data = json.loads(path.read_text())

            # 1) Summary global
            summary_text = self._generate_summary(data)
            summary_emb = self._get_embeddings([summary_text])[0]
            # TODO: subir summary_emb con metadatos {'cid': cid, 'section': 'summary'}

            # 2) Secciones temáticas
            for section, text in self._iter_sections(data):
                chunks = self._chunk_text(text)
                embeddings = self._get_embeddings(chunks)
                for chunk, emb in zip(chunks, embeddings):
                    # TODO: subir emb con metadatos {'cid': cid, 'section': section}
                    pass

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una búsqueda por embedding:
          - Obtiene embedding de la query.
          - Llama a Watsonx Vector DB con filtros.
          - Devuelve lista de hits agrupados por CID.
        """
        q_emb = self._get_embeddings([query])[0]
        # TODO: implementar llamada real al vector index:
        # response = self.vector_db.query(...)
        return []

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Llama al servicio de embeddings de Watsonx y devuelve la lista de vectores.
        """
        response = self.embed_service.create(inputs=texts)
        # Asumimos que `response.embeddings` es la lista de vectores devueltos
        return [emb.vector for emb in response.embeddings]

    def _iter_sections(self, data: Dict[str, Any]) -> List[Any]:
        """
        Genera pares (sección, texto) para las categorías de la molécula.
        """
        sections = ["structure", "safety", "spectra", "solubility", "search"]
        for sec in sections:
            text = json.dumps(data.get(sec, {}))
            yield sec, text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Trocea el texto en chunks de `chunk_size` tokens con `overlap`.
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
        Genera un resumen global de la molécula (40-70 palabras).
        Implementa aquí tu propia lógica o llamada a un modelo generativo.
        """
        name = data.get("names", {}).get("preferred", "Unknown")
        cid = data.get("search", {}).get("cid")
        return f"Molecule {name} (CID {cid}): summary placeholder."
