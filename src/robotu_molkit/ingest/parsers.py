# ingest/parsers.py
import logging
import re
import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# RDKit availability
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors
    RDKit_OK = True
except Exception as e:
    logging.warning("RDKit unavailable – chemical descriptors and fingerprints disabled (%s)", e)
    RDKit_OK = False

# ESOL model for logS
if RDKit_OK:
    def esol_logS(smiles: str) -> float:
        m = Chem.MolFromSmiles(smiles)
        logP = Descriptors.MolLogP(m)
        mw   = Descriptors.MolWt(m)
        rb   = Descriptors.NumRotatableBonds(m)
        ap   = sum(atom.GetIsAromatic() for atom in m.GetAtoms()) / m.GetNumHeavyAtoms()
        return 0.16 - 0.63*logP - 0.0062*mw + 0.066*rb - 0.74*ap

# pKa predictor disabled – not available via pip
PKA_OK = False



def find_section(sections: List[Dict[str, Any]], heading: str) -> Dict[str, Any]:
    for sec in sections:
        if sec.get("TOCHeading") == heading:
            return sec
        nested = find_section(sec.get("Section", []), heading)
        if nested:
            return nested
    return {}


def _extract_number(info: List[Dict[str, Any]], key: str) -> Optional[float]:
    for entry in info:
        if entry.get("Name") == key:
            num = entry.get("Value", {}).get("Number")
            if isinstance(num, dict):
                return num.get("Value")
    return None


def build_parsed(
    raw: Dict[str, Any],
    synonyms: Optional[Dict[str, Any]],
    props: Optional[Dict[str, Any]],
    view: Optional[Dict[str, Any]],
    cid: int,
    raw_path: Path,
) -> Dict[str, Any]:
    # --- Structure 3D ---
    xyz: Optional[List[Tuple[float, float, float]]] = None
    atom_symbols: Optional[List[str]] = None
    bond_orders = None
    if raw.get("Record"):
        sects = {s.get("TOCHeading"): s for s in raw["Record"].get("Section", [])}
        info3d = sects.get("3D Conformer", {}).get("Information", [])
        if info3d:
            c3d = info3d[0]["Value"]["Conformer3D"]
            xyz = [(c["X"], c["Y"], c["Z"]) for c in c3d.get("Coordinates", [])]
            atom_symbols = c3d.get("Atoms")
    elif raw.get("PC_Compounds"):
        comp = raw["PC_Compounds"][0]
        elems = comp.get("atoms", {}).get("element", [])
        coords_list = comp.get("coords", [])
        if coords_list:
            first = coords_list[0]
            confs = first.get("conformers", [])
            if confs:
                c = confs[0]
                xs, ys, zs = c.get("x", []), c.get("y", []), c.get("z", [])
                xyz = list(zip(xs, ys, zs))
                if RDKit_OK:
                    try:
                        ptable = Chem.GetPeriodicTable()
                        atom_symbols = [ptable.GetElementSymbol(n) for n in elems]
                    except Exception:
                        atom_symbols = [str(n) for n in elems]
                else:
                    atom_symbols = [str(n) for n in elems]
        aid1 = comp.get("bonds", {}).get("aid1", [])
        aid2 = comp.get("bonds", {}).get("aid2", [])
        orders = comp.get("bonds", {}).get("order", [])
        if aid1 and aid2 and orders:
            bond_orders = list(zip(aid1, aid2, orders))

    # --- Basic properties ---
    p = props.get("PropertyTable", {}).get("Properties", [{}])[0] if props else {}
    smiles = p.get("CanonicalSMILES")
    logp = p.get("XLogP")
    formal_charge = p.get("Charge")

    # --- Extended solubility ---
    logs = None
    # pKa prediction disabled
    pka_vals = None

    # --- PUG-View helper ---
    view_secs = view.get("Record", {}).get("Section", []) if view else []
    def info_fn(heading: str) -> List[Dict[str, Any]]:
        return find_section(view_secs, heading).get("Information", [])

    # --- Thermodynamics ---
    tinfo = info_fn("Thermodynamics")
    delta_h = _extract_number(tinfo, "Standard Enthalpy of Formation")
    entropy = _extract_number(tinfo, "Standard Molar Entropy")
    heat_capacity = _extract_number(tinfo, "Heat Capacity")

    # --- Safety (GHS codes) ---
    ghs_codes: List[str] = []
    for entry in info_fn("GHS Classification"):
        text_blob = ""
        val = entry.get("Value", {})
        for sm in val.get("StringWithMarkup", []):
            text_blob += sm.get("String", "") + " "
        s_val = val.get("String")
        if s_val:
            if isinstance(s_val, list):
                text_blob += " ".join(s_val)
            else:
                text_blob += str(s_val)
        codes = re.findall(r"\b[HP]\d{3}\b", text_blob)
        ghs_codes.extend(codes)
    seen = set()
    ghs_codes = [c for c in ghs_codes if not (c in seen or seen.add(c))]
    flash = _extract_number(info_fn("Physical Properties"), "Flash Point")
    ld50 = _extract_number(info_fn("Toxicity"), "LD50")

    # --- Spectral raw info ---
    spec = find_section(view_secs, "Spectral Information").get("Section", [])
    spectra_raw = {sub.get("TOCHeading"): sub.get("Information", []) for sub in spec}

    # --- Synonyms ---
    syns: List[str] = []
    if synonyms:
        syns = synonyms.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
    preferred = syns[0] if syns else None
    cas_like = next((s for s in syns if re.fullmatch(r"\d+-\d+-\d+", s)), None)

    # --- Fingerprints ---
    ecfp = maccs = None
    if RDKit_OK and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            # Use new MorganGenerator if available to avoid deprecation
            try:
                from rdkit.Chem.AllChem import MorganGenerator
                mg = MorganGenerator(radius=2)
                ecfp = mg.GetFingerprint(mol).ToBitString()
            except Exception:
                ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
            maccs = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        except Exception as e:
            logging.warning("Fingerprint error: %s", e)

    # --- Assemble final JSON ---
    return {
        "structure": {"xyz": xyz, "atom_symbols": atom_symbols, "bond_orders": bond_orders,
                       "formal_charge": formal_charge, "spin_multiplicity": None},
        "quantum": {k: None for k in ["h_core","g_two","mo_energies","homo_index",
                                        "mulliken_charges","esp_charges","dipole_moment","quadrupole_moment"]},
        "thermo": {"standard_enthalpy": delta_h, "entropy": entropy,
                    "heat_capacity": heat_capacity},
        "safety": {"ghs_codes": ghs_codes, "flash_point": flash, "ld50": ld50},
        "spectra": {"raw": spectra_raw},
        "solubility": {"logp": logp, "logs": logs, "pka": pka_vals},
        "search": {"cid": cid, "inchi": p.get("InChI"), "inchikey": p.get("InChIKey"),
                    "smiles": smiles, "ecfp": ecfp, "maccs": maccs,
                    "embeddings": {"summary": None, "structure": None}},
        "names": {"preferred": preferred, "cas_like": cas_like,
                  "systematic": None, "traditional": None, "synonyms": syns},
        "meta": {"fetched": datetime.datetime.utcnow().isoformat() + "Z",
                  "source": "PubChem",
                  "source_version": raw.get("Record", {}).get("RecordMetadata", {}).get("ReleaseDate"),
                  "cache_path": str(raw_path)},
    }
