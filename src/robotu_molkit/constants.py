from pathlib import Path

MAX_RPS = 5
MAX_RPM = 400
TIMEOUT_S = 30
DEFAULT_RAW_DIR = Path("data/downloaded_data")
DEFAULT_PARSED_DIR = Path("data/parsed")
DEFAULT_CONCURRENCY = 5

# Lista de endpoints
RECORD_API   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/JSON?record_type=3d"
SYNONYMS_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
PROPERTIES_API = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/"
    "CanonicalSMILES,InChI,InChIKey,XLogP,Charge/JSON"
)
PUG_VIEW_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
EMBED_MODEL_ID = "ibm/slate-30m-english-rtrvr"