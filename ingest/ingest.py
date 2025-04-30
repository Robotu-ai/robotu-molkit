#!/usr/bin/env python3
"""
ingest.py – PubChem → RobotU JSON (IBM embeddings, PUG-View)
==============================================================
* Robust to NumPy-2 / RDKit ABI breakages (fingerprints skipped if RDKit import fails).
* Ignores non-numeric tokens in the CID list (e.g. stray words).
* Pulls 3-D record, synonyms, basic properties; thermo parsed from PUG-View.
* Structure embeddings via IBM watsonx.ai (Granite 30 M). Summary embedding left null.
* Raw JSON saved to **data/downloaded_data/**, parsed payloads to **data/parsed/**.
* Respects PubChem rate-limits (5 RPS · 400 RPM).

Examples
--------
```bash
python ingest.py 2519 5957 \
  --api-key $IBM_API_KEY \
  --project-id $IBM_PROJECT_ID
```

Dependencies
------------
```bash
pip install aiohttp aiolimiter tqdm ibm-watsonx-ai  # rdkit-pypi optional
```
"""

from __future__ import annotations
import argparse
import asyncio
import datetime
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientResponseError
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm

# RDKit (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    RDKit_OK = True
except Exception as e:
    logging.warning("RDKit unavailable – fingerprints disabled (%s)", e)
    RDKit_OK = False

# IBM embeddings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
EMBED_MODEL_ID = "ibm/granite-embedding-30m-english"

# Endpoints PubChem
RECORD_API     = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/JSON?record_type=3d"
SYNONYMS_API   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
PROPERTIES_API = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/"
    "CanonicalSMILES,InChI,InChIKey,XLogP,Charge/JSON"
)
PUG_VIEW_API   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"

# Rate limits and defaults
MAX_RPS = 5
MAX_RPM = 400
TIMEOUT_S = 30
DEFAULT_RAW_DIR = Path("data/downloaded_data")
DEFAULT_PARSED_DIR = Path("data/parsed")
DEFAULT_CONCURRENCY = 5

async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    sec_limiter: AsyncLimiter,
    min_limiter: AsyncLimiter,
) -> Optional[Dict[str, Any]]:
    try:
        async with sec_limiter, min_limiter:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()
    except ClientResponseError as e:
        logging.warning("HTTP %s for %s", e.status, url)
    except Exception as e:
        logging.warning("Request error %s: %s", url, e)
    return None

# Recursively locate PUG-View sections by heading

def find_section(sections: List[Dict[str, Any]], heading: str) -> Dict[str, Any]:
    for sec in sections:
        if sec.get("TOCHeading") == heading:
            return sec
        nested = find_section(sec.get("Section", []), heading)
        if nested:
            return nested
    return {}

# Extract numeric values from PUG-View entries

def _extract_number(info: List[Dict[str, Any]], key: str) -> Optional[float]:
    for entry in info:
        if entry.get("Name") == key:
            num = entry.get("Value", {}).get("Number")
            if isinstance(num, dict):
                return num.get("Value")
    return None

# Assemble full parsed JSON

def build_parsed(
    raw: Dict[str, Any],
    synonyms: Optional[Dict[str, Any]],
    props: Optional[Dict[str, Any]],
    view: Optional[Dict[str, Any]],
    embed_service: Embeddings,
    cid: int,
    raw_path: Path,
) -> Dict[str, Any]:
    # Parse 3-D structure
    record = raw.get("Record", {})
    sects = {s.get("TOCHeading"): s for s in record.get("Section", [])}
    xyz = atom_symbols = bond_orders = None
    info3d = sects.get("3D Conformer", {}).get("Information", [])
    if info3d:
        c3d = info3d[0]["Value"]["Conformer3D"]
        xyz = [(c["X"], c["Y"], c["Z"]) for c in c3d.get("Coordinates", [])]
        atom_symbols = c3d.get("Atoms")

    # Basic properties
    p = props.get("PropertyTable", {}).get("Properties", [{}])[0] if props else {}
    smiles = p.get("CanonicalSMILES")
    logp = p.get("XLogP")
    formal_charge = p.get("Charge")

    # PUG-View sections
    view_secs = view.get("Record", {}).get("Section", []) if view else []
    def info(heading: str) -> List[Dict[str, Any]]:
        return find_section(view_secs, heading).get("Information", [])

    # Thermodynamics
    tinfo = info("Thermodynamics")
    delta_h = _extract_number(tinfo, "Standard Enthalpy of Formation")
    entropy = _extract_number(tinfo, "Standard Molar Entropy")
    heat_capacity = _extract_number(tinfo, "Heat Capacity")

    # Safety and toxicity
    ghs = [e.get("Name") for e in info("GHS Classification")]
    flash = _extract_number(info("Physical Properties"), "Flash Point")
    ld50 = _extract_number(info("Toxicity"), "LD50")

    # Spectral raw info
    spec = find_section(view_secs, "Spectral Information").get("Section", [])
    spectra_raw = {sub.get("TOCHeading"): sub.get("Information", []) for sub in spec}

    # Synonyms
    syns = []
    if synonyms:
        syns = synonyms.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
    preferred = syns[0] if syns else None
    cas_like = next((s for s in syns if re.fullmatch(r"\d+-\d+-\d+", s)), None)

    # Fingerprints
    ecfp = maccs = None
    if RDKit_OK and smiles:
        try:
            m = Chem.MolFromSmiles(smiles)
            ecfp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048).ToBitString()
            maccs = MACCSkeys.GenMACCSKeys(m).ToBitString()
        except Exception as e:
            logging.warning("Fingerprint error: %s", e)

    # Embeddings
    struct_emb = None
    if smiles:
        try:
            struct_emb = embed_service.embed_documents(texts=[smiles])[0]
        except Exception as e:
            logging.warning("Embedding error: %s", e)

    return {
        "structure": {
            "xyz": xyz,
            "atom_symbols": atom_symbols,
            "bond_orders": bond_orders,
            "formal_charge": formal_charge,
            "spin_multiplicity": None,
        },
        "quantum": {k: None for k in [
            "h_core","g_two","mo_energies","homo_index",
            "mulliken_charges","esp_charges","dipole_moment","quadrupole_moment"
        ]},
        "thermo": {
            "standard_enthalpy": delta_h,
            "entropy": entropy,
            "heat_capacity": heat_capacity,
        },
        "safety": {
            "ghs_codes": ghs,
            "flash_point": flash,
            "ld50": ld50,
        },
        "spectra": {"raw": spectra_raw},
        "solubility": {"logp": logp, "pka": None},
        "search": {
            "cid": cid,
            "inchi": p.get("InChI"),
            "inchikey": p.get("InChIKey"),
            "smiles": smiles,
            "ecfp": ecfp,
            "maccs": maccs,
            "embeddings": {"summary": None, "structure": struct_emb},
        },
        "names": {
            "preferred": preferred,
            "cas_like": cas_like,
            "systematic": None,
            "traditional": None,
            "synonyms": syns,
        },
        "meta": {
            "fetched": datetime.datetime.utcnow().isoformat() + "Z",
            "source": "PubChem",
            "source_version": record.get("RecordMetadata", {}).get("ReleaseDate"),
            "cache_path": str(raw_path),
        },
    }

async def process_cid(
    cid: str,
    session: aiohttp.ClientSession,
    raw_dir: Path,
    parsed_dir: Path,
    sec_limiter: AsyncLimiter,
    min_limiter: AsyncLimiter,
    embed_service: Embeddings,
) -> None:
    raw_path = raw_dir / f"pubchem_{cid}_raw.json"
    parsed_path = parsed_dir / f"pubchem_{cid}.json"

    raw = await _get_json(session, RECORD_API.format(cid=cid), sec_limiter, min_limiter)
    if not raw:
        return
    raw_path.write_text(json.dumps(raw, indent=2))

    syn = await _get_json(session, SYNONYMS_API.format(cid=cid), sec_limiter, min_limiter)
    props = await _get_json(session, PROPERTIES_API.format(cid=cid), sec_limiter, min_limiter)
    view = await _get_json(session, PUG_VIEW_API.format(cid=cid), sec_limiter, min_limiter)

    parsed = build_parsed(raw, syn, props, view, embed_service, int(cid), raw_path)
    parsed_path.write_text(json.dumps(parsed, indent=2))

async def worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    raw_dir: Path,
    parsed_dir: Path,
    sec_limiter: AsyncLimiter,
    min_limiter: AsyncLimiter,
    embed_service: Embeddings,
) -> None:
    while True:
        cid = await queue.get()
        if cid is None:
            queue.task_done()
            break
        try:
            await process_cid(cid, session, raw_dir, parsed_dir, sec_limiter, min_limiter, embed_service)
        except Exception as e:
            logging.error("Error processing CID %s: %s", cid, e)
        finally:
            queue.task_done()

async def run(
    cids: List[str],
    raw_dir: Path,
    parsed_dir: Path,
    concurrency: int,
    embed_service: Embeddings,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    queue: asyncio.Queue = asyncio.Queue()
    for cid in cids:
        if re.fullmatch(r"\d+", cid):
            queue.put_nowait(cid)
    for _ in range(concurrency):
        queue.put_nowait(None)

    sec_limiter = AsyncLimiter(MAX_RPS, 1)
    min_limiter = AsyncLimiter(MAX_RPM, 60)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(worker(
            queue, session, raw_dir, parsed_dir, sec_limiter, min_limiter, embed_service
        )) for _ in range(concurrency)]
        await queue.join()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

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