#!/usr/bin/env python3
"""
ingest.py  – PubChem‑safe downloader
===========================================
Step 1 of the RobotU Molkit pipeline downloads **raw PubChem JSON** while **respecting the official
rate‑limits**:

* **≤ 5 requests / second**
* **≤ 400 requests / minute**
* **30 s individual request time‑out**

The script uses two token‑bucket limiters so you can raise concurrency without violating
PubChem’s usage policy.

Usage examples
--------------
```bash
# direct list of CIDs
python ingest.py 2519 2244 5957

# newline‑delimited file of CIDs, custom output dir, 12 workers
python ingest.py --file cids.txt --out raw --concurrency 12
```

Dependencies
------------
```bash
pip install aiohttp aiolimiter tqdm
```
* **aiohttp** ≥ 3.9 ….. async HTTP
* **aiolimiter** ….. token‑bucket rate‑limiting (RFC‑friendly)
* **tqdm** ….. pretty progress bar (optional)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List

import aiohttp
from aiohttp import ClientResponseError
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm

API_TEMPLATE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"

# ---------- PubChem limits ---------------------------------------------------
MAX_RPS = 5           # 5 per second
MAX_RPM = 400         # 400 per minute

DEFAULT_CONCURRENCY = 5  # keep at/below RPS by default
REQUEST_TIMEOUT = 30     # seconds (PubChem hard‑limit 30 s)


# ----------------------------------------------------------------------------
# Async helpers
# ----------------------------------------------------------------------------

def chunked(iterable: Iterable[str], n: int) -> Iterable[List[str]]:
    """Yield *n*-sized chunks from *iterable*."""
    it = iter(iterable)
    while True:
        chunk = list([next(it) for _ in range(n) if True])  # noqa: B007 – keep simple
        if not chunk:
            break
        yield chunk


async def fetch(cid: str, session: aiohttp.ClientSession, out_dir: Path) -> None:
    """Fetch CID JSON and write it to *out_dir/<cid>.json*."""
    url = API_TEMPLATE.format(cid=cid)
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except ClientResponseError as e:
        logging.warning("CID %s – HTTP %s (%s)", cid, e.status, e.message)
        raise
    except Exception as e:  # pylint: disable=broad-except
        logging.warning("CID %s – %s", cid, e)
        raise
    (out_dir / f"{cid}.json").write_text(json.dumps(data, indent=2))


async def worker(queue: asyncio.Queue[str], session: aiohttp.ClientSession, out_dir: Path,
                 sec_limiter: AsyncLimiter, min_limiter: AsyncLimiter) -> None:
    while True:
        cid = await queue.get()
        if cid is None:
            queue.task_done(); break  # sentinel for shutdown
        try:
            async with sec_limiter:
                async with min_limiter:
                    await fetch(cid, session, out_dir)
        except Exception:  # errors already logged
            pass
        queue.task_done()


async def run(cids: List[str], out_dir: Path, concurrency: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build queues & limiters
    queue: asyncio.Queue[str] = asyncio.Queue()
    for cid in cids:
        queue.put_nowait(cid)
    for _ in range(concurrency):
        queue.put_nowait(None)  # shutdown sentinels

    sec_limiter = AsyncLimiter(MAX_RPS, 1)
    min_limiter = AsyncLimiter(MAX_RPM, 60)

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(worker(queue, session, out_dir, sec_limiter, min_limiter))
                 for _ in range(concurrency)]
        with tqdm(total=len(cids), desc="Downloading", unit="mol") as pbar:
            while not queue.empty():
                downloaded = len(list(out_dir.glob("*.json")))
                pbar.n = downloaded
                pbar.refresh()
                await asyncio.sleep(0.5)
        await queue.join()
        for t in tasks:
            t.cancel()


# ----------------------------------------------------------------------------
# CLI entry‑point
# ----------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download raw PubChem JSON by CID (rate‑limit safe).")
    p.add_argument("cids", nargs="*", help="CID(s) to download (int)")
    p.add_argument("--file", "-f", help="Text file with one CID per line")
    p.add_argument("--out", "-o", default="raw", help="Output directory [default: ./raw]")
    p.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY,
                   help="Number of concurrent tasks (should not exceed 5 unless you change RPS)")
    return p.parse_args()


def main() -> None:
    args = parse_cli()

    # Resolve list of CIDs
    if args.file:
        cids = [line.strip() for line in Path(args.file).read_text().splitlines() if line.strip()]
    else:
        cids = args.cids
    if not cids:
        print("❌  Provide CID(s) via arguments or --file", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("▶ Starting download of %d molecule(s) …", len(cids))

    try:
        asyncio.run(run(cids, out_dir, args.concurrency))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user – partial downloads kept.")

    logging.info("✔ Done. JSON written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
