# ingest/workers.py
import asyncio, json, logging
from pathlib import Path
from typing import List
from aiolimiter import AsyncLimiter
import aiohttp

from robotu_molkit.constants import MAX_RPS, MAX_RPM, TIMEOUT_S, DEFAULT_RAW_DIR, DEFAULT_PARSED_DIR

from .api_clients import fetch_record, fetch_synonyms, fetch_properties, fetch_view
from .parsers import build_parsed
from ibm_watsonx_ai.foundation_models import Embeddings
import re

async def process_cid(
    cid: int,
    session: aiohttp.ClientSession,
    raw_dir: Path,
    parsed_dir: Path,
    sec_limiter: AsyncLimiter,
    min_limiter: AsyncLimiter,
) -> None:
    raw_path = raw_dir / f"pubchem_{cid}_raw.json"
    parsed_path = parsed_dir / f"pubchem_{cid}.json"

    raw = await fetch_record(cid, session, sec_limiter, min_limiter)
    if not raw:
        return
    raw_path.write_text(json.dumps(raw, indent=2))

    syn  = await fetch_synonyms(cid, session, sec_limiter, min_limiter)
    props= await fetch_properties(cid, session, sec_limiter, min_limiter)
    view = await fetch_view(cid, session, sec_limiter, min_limiter)

    parsed = build_parsed(raw, syn, props, view, int(cid), raw_path)
    parsed_path.write_text(json.dumps(parsed, indent=2))

async def worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    raw_dir: Path,
    parsed_dir: Path,
    sec_limiter: AsyncLimiter,
    min_limiter: AsyncLimiter,
) -> None:
    while True:
        cid = await queue.get()
        if cid is None:
            queue.task_done()
            break
        try:
            await process_cid(cid, session, raw_dir, parsed_dir, sec_limiter, min_limiter)
        except Exception as e:
            logging.error("Error processing CID %s: %s", cid, e)
        finally:
            queue.task_done()

async def run(
    cids: List[int],
    raw_dir: Path,
    parsed_dir: Path,
    concurrency: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    queue: asyncio.Queue = asyncio.Queue()
    for cid in cids:
        queue.put_nowait(cid)
    for _ in range(concurrency):
        queue.put_nowait(None)

    sec_limiter = AsyncLimiter(MAX_RPS, 1)
    min_limiter = AsyncLimiter(MAX_RPM, 60)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(worker(
            queue, session, raw_dir, parsed_dir, sec_limiter, min_limiter
        )) for _ in range(concurrency)]
        await queue.join()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)