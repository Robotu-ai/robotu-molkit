import typer
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings

from robotu_molkit.constants import (
    DEFAULT_RAW_DIR,
    DEFAULT_PARSED_DIR,
    DEFAULT_CONCURRENCY,
    EMBED_MODEL_ID,
)
from robotu_molkit.ingest.workers import run as _run_workers
from robotu_molkit.config import load_credentials
from robotu_molkit.vector.watsonx_index import WatsonxIndex

CONFIG_PATH = Path.home() / ".config" / "molkit" / "config.json"

# CLI principal con descripción general
desc = "Download and parse molecules from PubChem."
app = typer.Typer(help=desc, add_completion=False)

@app.command("config")
def config(
    api_key: str = typer.Option(..., "--api-key", "-k", help="IBM API Key"),
    project_id: str = typer.Option(..., "--project-id", "-j", help="IBM Project ID"),
):
    """
    Guarda las credenciales IBM en ~/.config/molkit/config.json
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "api_key": api_key,
        "project_id": project_id
    }, indent=2))
    typer.secho(f"Credenciales guardadas en {CONFIG_PATH}", fg=typer.colors.GREEN)

@app.command("ingest")
def ingest(
    cids: list[int] = typer.Argument(..., help="CID(s) to fetch"),
    file: Path = typer.Option(None, "--file", "-f", help="File with one CID per line"),
    raw_dir: Path = typer.Option(DEFAULT_RAW_DIR, "--raw-dir", "-r"),
    parsed_dir: Path = typer.Option(DEFAULT_PARSED_DIR, "--parsed-dir", "-p"),
    concurrency: int = typer.Option(DEFAULT_CONCURRENCY, "--concurrency", "-c"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k",
        help="IBM API Key (override config/envvar)"
    ),
    project_id: Optional[str] = typer.Option(
        None, "--project-id", "-j",
        help="IBM Project ID (override config/envvar)"
    ),
    ibm_url: str = typer.Option(
        "https://us-south.ml.cloud.ibm.com", "--ibm-url", help="IBM service URL"
    ),
):
    """
    Fetches CID(s) from PubChem, saves raw JSON and parsed Molecule payloads, y genera embeddings con watsonx.ai.
    """
    saved_key, saved_proj = load_credentials()
    api_key = api_key or saved_key
    project_id = project_id or saved_proj

    if not api_key or not project_id:
        typer.secho(
            "❌ Faltan credenciales IBM: pásalas con --api-key/--project-id, "
            "o bien ejecuta `molkit config` o define las envvars IBM_API_KEY / IBM_PROJECT_ID.",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(code=1)

    if file:
        file_cids = [int(line.strip()) for line in file.read_text().splitlines() if line.strip()]
        cids = file_cids + cids
    if not cids:
        typer.secho("❌ No CIDs provided", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    creds = Credentials(api_key=api_key, url=ibm_url)
    embed_service = Embeddings(
        model_id=EMBED_MODEL_ID,
        credentials=creds,
        project_id=project_id,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting ingest of %d CIDs...", len(cids))
    try:
        asyncio.run(_run_workers(cids, raw_dir, parsed_dir, concurrency, embed_service))
    except KeyboardInterrupt:
        typer.secho("⚠️ Ingest interrupted by user", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    logging.info("Done! Raw → %s | Parsed → %s", raw_dir, parsed_dir)

@app.command("embed")
def embed_command(
    cids: list[int] = typer.Option(..., "--cids", "-c", help="Lista de CIDs a procesar."),
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
    api_key, project_id = load_credentials(api_key, project_id)
    if not api_key or not project_id:
        typer.secho(
            "❌ Faltan credenciales IBM: pásalas con --api-key/--project-id, "
            "o configura ~/.config/molkit/config.json",
            err=True,
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()

