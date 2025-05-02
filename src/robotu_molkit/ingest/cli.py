# src/robotu_molkit/ingest/cli.py
import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Optional

import typer
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings

from .utils import DEFAULT_RAW_DIR, DEFAULT_PARSED_DIR, DEFAULT_CONCURRENCY, EMBED_MODEL_ID
from .workers import run as _run_workers
from robotu_molkit.utils import load_credentials
from robotu_molkit.vector import WatsonxIndex

ingest_app = typer.Typer(help="Download and parse molecules from PubChem.")
CONFIG_PATH = Path.home() / ".config" / "molkit" / "config.json"

@ingest_app.command("run")
def run_ingest(
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
    # Cargar credenciales guardadas
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

    # Combinar CIDs de archivo y argumentos CLI
    if file:
        file_cids = [int(line.strip()) for line in file.read_text().splitlines() if line.strip()]
        cids = file_cids + cids
    if not cids:
        typer.secho("❌ No CIDs provided", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Configurar servicio de embeddings
    creds = Credentials(api_key=api_key, url=ibm_url)
    embed_service = Embeddings(
        model_id=EMBED_MODEL_ID,
        credentials=creds,
        project_id=project_id,
    )

    # Ejecutar ingest y embeddings
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting ingest of %d CIDs...", len(cids))
    try:
        asyncio.run(_run_workers(cids, raw_dir, parsed_dir, concurrency, embed_service))
    except KeyboardInterrupt:
        typer.secho("⚠️ Ingest interrupted by user", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    logging.info("Done! Raw → %s | Parsed → %s", raw_dir, parsed_dir)

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