# cli/ingest.py

import typer
from typing import Optional, List
from molkit.vector import WatsonxIndex

from .utils import load_credentials

app = typer.Typer(help="Descarga y procesa moléculas de PubChem y genera embeddings.")

@app.command("embed")
def embed(
    cids: List[int] = typer.Option(..., "--cids", "-c", help="Lista de CIDs a procesar."),
    model: str = typer.Option(
        "granite-embedding-278m-multilingual", "--model", "-m",
        help="Modelo Granite a usar para embeddings."
    ),
    chunk_size: int = typer.Option(250, "--chunk-size", help="Tamaño de chunk en tokens."),
    overlap: int = typer.Option(40, "--overlap", help="Tokens de superposición entre chunks."),
    fast: bool = typer.Option(False, "--fast", help="Alias para usar granite-embedding-107m-multilingual."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="IBM API Key (override config/envvar)."
    ),
    project_id: Optional[str] = typer.Option(
        None, "--project-id", "-j", help="IBM Project ID (override config/envvar)."
    ),
    ibm_url: str = typer.Option(
        "https://us-south.ml.cloud.ibm.com", "--ibm-url", help="IBM service URL."
    ),
):
    """
    Genera resúmenes y embeddings con watsonx.ai para los CIDs indicados.
    """
    # Obtener credenciales guardadas o env (moved to utils)
    api_key, project_id = load_credentials(api_key, project_id)
    if not api_key or not project_id:
        typer.secho(
            "❌ Faltan credenciales IBM: pásalas con --api-key/--project-id, "
            "o configura ~/.config/molkit/config.json",
            err=True,
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Selección rápida de modelo
    if fast:
        model = "granite-embedding-107m-multilingual"

    typer.echo(f"Usando modelo {model}, chunk_size={chunk_size}, overlap={overlap}")
    index = WatsonxIndex(
        api_key=api_key,
        project_id=project_id,
        model=model,
        chunk_size=chunk_size,
        overlap=overlap,
        ibm_url=ibm_url
    )
    index.ingest_cids(cids)
    typer.secho("✅ Embeddings generados y subidos a Watsonx Vector DB.", fg=typer.colors.GREEN)
