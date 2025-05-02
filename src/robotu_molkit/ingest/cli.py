# src/robotu_molkit/ingest/cli.py
import asyncio
import logging
from pathlib import Path

import typer
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings

from .utils import DEFAULT_RAW_DIR, DEFAULT_PARSED_DIR, DEFAULT_CONCURRENCY, EMBED_MODEL_ID
from .workers import run as _run_workers

ingest_app = typer.Typer(help="Download and parse molecules from PubChem.")

@ingest_app.command("run")
def run_ingest(
    cids: list[int] = typer.Argument(..., help="CID(s) to fetch"),
    file: Path = typer.Option(None, "--file", "-f", help="File with one CID per line"),
    raw_dir: Path = typer.Option(DEFAULT_RAW_DIR, "--raw-dir", "-r"),
    parsed_dir: Path = typer.Option(DEFAULT_PARSED_DIR, "--parsed-dir", "-p"),
    concurrency: int = typer.Option(DEFAULT_CONCURRENCY, "--concurrency", "-c"),
    api_key: str = typer.Option(..., "--api-key", "-k", envvar="IBM_API_KEY", help="IBM API Key"),
    project_id: str = typer.Option(..., "--project-id", "-j", envvar="IBM_PROJECT_ID", help="IBM Project ID"),
    ibm_url: str = typer.Option("https://us-south.ml.cloud.ibm.com", "--ibm-url", help="IBM service URL"),
):
    """
    Fetches CID(s) from PubChem, saves raw JSON and parsed Molecule payloads.
    """
    # 1) combinar CIDs de --file
    if file:
        file_cids = [int(l) for l in file.read_text().splitlines() if l.strip()]
        cids = file_cids + cids
    if not cids:
        typer.secho("❌ No CIDs provided", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 2) preparar servicio de embeddings
    creds = Credentials(api_key=api_key, url=ibm_url)
    embed_service = Embeddings(
        model_id=EMBED_MODEL_ID,
        credentials=creds,
        project_id=project_id,
    )

    # 3) ejecutar ingest
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting ingest of %d CIDs...", len(cids))
    try:
        asyncio.run(_run_workers(cids, raw_dir, parsed_dir, concurrency, embed_service))
    except KeyboardInterrupt:
        typer.secho("⚠️ Ingest interrupted by user", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    logging.info("Done! Raw → %s | Parsed → %s", raw_dir, parsed_dir)
