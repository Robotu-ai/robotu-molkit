# ingest/cli.py
import argparse, asyncio, sys, logging
from pathlib import Path
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings

from .utils import DEFAULT_RAW_DIR, DEFAULT_PARSED_DIR, DEFAULT_CONCURRENCY, EMBED_MODEL_ID
from .workers import run

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PubChem ingest → RobotU JSON pipeline")
    parser.add_argument("cids", nargs="*", help="CID(s) to fetch (int)")
    parser.add_argument("--file", "-f", help="File with one CID per line")
    parser.add_argument("--raw-dir", "-r", default=DEFAULT_RAW_DIR, help="Raw JSON dir")
    parser.add_argument("--parsed-dir", "-p", default=DEFAULT_PARSED_DIR, help="Parsed JSON dir")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY, help="Workers")
    parser.add_argument("--api-key", "-k", dest="api_key", required=True, help="IBM API Key")
    parser.add_argument("--project-id", "-j", dest="project_id", required=True, help="IBM Project ID")
    parser.add_argument("--ibm-url", default="https://us-south.ml.cloud.ibm.com", help="IBM URL")
    return parser.parse_args()

def main() -> None:
    args = parse_cli()
    if args.file:
        cids = [line.strip() for line in Path(args.file).read_text().splitlines()]
    else:
        cids = args.cids or []
    if not cids:
        print("❌ No CIDs provided.", file=sys.stderr)
        sys.exit(1)

    creds = Credentials(api_key=args.api_key, url=args.ibm_url)
    embed_service = Embeddings(
        model_id=EMBED_MODEL_ID,
        credentials=creds,
        project_id=args.project_id,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting ingest of %d CIDs...", len(cids))
    try:
        asyncio.run(run(
            cids,
            Path(args.raw_dir),
            Path(args.parsed_dir),
            args.concurrency,
            embed_service,
        ))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
    logging.info("Done: raw in %s, parsed in %s", args.raw_dir, args.parsed_dir)

if __name__ == "__main__":
    main()