import typer
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, List

from robotu_molkit.constants import (
    DEFAULT_RAW_DIR,
    DEFAULT_PARSED_DIR,
    DEFAULT_CONCURRENCY,
    DEFAULT_EMBED_MODEL_ID,
    DEFAULT_WATSONX_AI_URL,
    FAST_EMBED_MODEL_ID
)
from robotu_molkit.ingest.workers import run as _run_workers
from robotu_molkit.config import load_credentials
from robotu_molkit.vector.watsonx_index import WatsonxIndex

CONFIG_PATH = Path.home() / ".config" / "molkit" / "config.json"

# Main CLI with overall description
desc = "Download and parse molecules from PubChem."
app = typer.Typer(help=desc, add_completion=False)

@app.command("config")
def config(
    api_key: str = typer.Option(..., "--watsonx-api-key", "-k", help="IBM Watsonx API Key"),
    project_id: str = typer.Option(..., "--watsonx-project-id", "-j", help="IBM Watsonx Project ID"),
):
    """
    Save IBM Watsonx credentials to ~/.config/molkit/config.json
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "api_key": api_key,
        "project_id": project_id
    }, indent=2))
    typer.secho(f"Credentials saved to {CONFIG_PATH}", fg=typer.colors.GREEN)

@app.command("ingest")
def ingest(
    cids: Optional[List[int]] = typer.Argument(None, help="CID(s) to fetch"),
    file: Path = typer.Option(None, "--file", "-f", help="File with one CID per line"),
    raw_dir: Path = typer.Option(DEFAULT_RAW_DIR, "--raw-dir", "-r", help="Directory to save raw JSON files"),
    parsed_dir: Path = typer.Option(DEFAULT_PARSED_DIR, "--parsed-dir", "-p", help="Directory to save parsed payloads"),
    concurrency: int = typer.Option(DEFAULT_CONCURRENCY, "--concurrency", "-c", help="Number of concurrent workers"),
  ):
    """
    Fetch CID(s) from PubChem, save raw JSON and parsed Molecule payloads.
    """
    cids = cids or []
    if file:
        file_cids = [int(line.strip()) for line in file.read_text().splitlines() if line.strip()]
        cids = file_cids + cids
    if not cids:
        typer.secho("❌ No CIDs provided", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting ingest of %d CIDs...", len(cids))
    try:
        asyncio.run(_run_workers(cids, raw_dir, parsed_dir, concurrency))
    except KeyboardInterrupt:
        typer.secho("⚠️ Ingest interrupted by user", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    logging.info("Done! Raw → %s | Parsed → %s", raw_dir, parsed_dir)

@app.command("embed")
def embed_command(
    parsed_dir: Path = typer.Option(
        "data/parsed", "--parsed-dir", "-p",
        help="Folder with parsed PubChem JSON files (e.g. pubchem_*.json)."
    ),
    out_dir: Path = typer.Option(
        "data/vectors", "--out-dir", "-o",
        help="Destination folder for watsonx_vectors.jsonl."
    ),
    model: str = typer.Option(
        DEFAULT_EMBED_MODEL_ID, "--model", "-m",
        help="Granite embedding model id."
    ),
    fast: bool = typer.Option(
        False, "--fast",
        help="Shortcut for granite-embedding-107m-multilingual."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--watsonx-api-key", "-k",
        help="IBM Watsonx API Key (override config/env var)."
    ),
    project_id: Optional[str] = typer.Option(
        None, "--watsonx-project-id", "-j",
        help="IBM Watsonx Project ID (override config/env var)."
    ),
    ibm_url: str = typer.Option(
        DEFAULT_WATSONX_AI_URL, "--watsonx-url",
        help="IBM Watsonx inference URL."
    ),
):
    """
    Generate a single “general” summary and embedding for **every** parsed
    PubChem JSON in PARSED_DIR and save them to a JSONL file that can later be
    bulk‑uploaded to Watsonx Vector DB.
    """
    api_key, project_id = load_credentials(api_key, project_id)
    if not api_key or not project_id:
        typer.secho(
            "❌  Missing IBM Watsonx credentials.  Pass them via "
            "--watsonx-api-key / --watsonx-project-id or run `molkit config`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if fast:
        model = FAST_EMBED_MODEL_ID

    typer.echo(f"Embedding model: {model}")
    idx = WatsonxIndex(
        api_key=api_key,
        project_id=project_id,
        model=model,
        ibm_url=ibm_url,
    )

    jsonl_path = idx.ingest_folder(parsed_dir=parsed_dir, out_dir=out_dir)
    typer.secho(f"✅  Embeddings written to {jsonl_path}", fg=typer.colors.GREEN)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
