# ingest/api_clients.py
import aiohttp
import logging
from typing import Any, Dict, Optional
from aiolimiter import AsyncLimiter

from robotu_molkit.constants import RECORD_API, SYNONYMS_API, PROPERTIES_API, PUG_VIEW_API, MAX_RPS, MAX_RPM, TIMEOUT_S

async def _get_json(
    session: aiohttp.ClientSession, url: str,
    sec_lim: AsyncLimiter, min_lim: AsyncLimiter
) -> Optional[Dict[str, Any]]:
    try:
        async with sec_lim, min_lim:
            async with session.get(url) as r:
                r.raise_for_status()
                return await r.json()
    except Exception as e:
        logging.warning("Error fetching %s: %s", url, e)
        return None

async def fetch_record(cid: str, session, sec_lim, min_lim):
    return await _get_json(session, RECORD_API.format(cid=cid), sec_lim, min_lim)

async def fetch_synonyms(cid: str, session, sec_lim, min_lim):
    return await _get_json(session, SYNONYMS_API.format(cid=cid), sec_lim, min_lim)

async def fetch_properties(cid: str, session, sec_lim, min_lim):
    return await _get_json(session, PROPERTIES_API.format(cid=cid), sec_lim, min_lim)

async def fetch_view(cid: str, session, sec_lim, min_lim):
    return await _get_json(session, PUG_VIEW_API.format(cid=cid), sec_lim, min_lim)
