# ingest/parsers.py
import logging
import re, json
import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from collections import OrderedDict
import requests

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

#–––  regex  ––––––––––––––––––––––––––––––––––––––––––
# Hnnn followed by a required percentage, e.g. “H302 (98.9%)”
_H_RX = re.compile(r"\b(H\d{3})(?!\+)\s*\((\d+(?:\.\d+)?)%\)")

def extract_h_codes(view_secs: List[Dict[str, Any]], min_pct: float = 10.0) -> List[str]:
    """
    Return a sorted list of unique GHS H‑codes whose notification percentage
    is >= min_pct.  P‑codes and other entries are ignored entirely.
    """
    ghs = find_section(view_secs, "GHS Classification")
    if not ghs:
        return []

    hazard: set[str] = set()

    for info in ghs.get("Information", []):
        if info.get("Name") != "GHS Hazard Statements":
            continue

        text = " ".join(
            sm.get("String", "")
            for sm in info.get("Value", {}).get("StringWithMarkup", [])
        )

        for h_code, pct in _H_RX.findall(text):
            if float(pct) >= min_pct:
                hazard.add(h_code)

    return sorted(hazard)

# ---------------------------------------------------
# Ontoloty extraction
# ---------------------------------------------------
def _collect_strings(value: Dict[str, Any]) -> List[str]:
    """
    Devuelve una lista con todos los textos que aparezcan en
    Value → StringWithMarkup y/o String.
    """
    out: List[str] = []

    # StringWithMarkup  (lista de dicts)
    for sm in value.get("StringWithMarkup", []):
        s = sm.get("String")
        if s:
            out.append(s)

    # String  (puede ser str o lista[str])
    if "String" in value:
        s = value["String"]
        out.extend(s if isinstance(s, list) else [s])

    return out

def _clean_term(term: str) -> str:
    term = term.lower().strip()

    # 1) quita paréntesis y lo que hay después
    term = term.split('(')[0]

    # 2) elimina la cláusula "in which …"
    term = re.split(r"\bin which\b", term)[0]

    # 3) corta a la izquierda de " that ", " which ", " with "
    term = re.split(r"\b(?:that|which|with)\b", term)[0]

    # 4) elimina conectores al final
    term = re.sub(r"\b(and|or)$", "", term).strip(",; .")

    return term

def _walk_information(node):
    """Generador recursivo que rinde todos los bloques 'Information'."""
    if isinstance(node, dict):
        if node.get("Information"):
            for inf in node["Information"]:
                yield inf
        for sub in node.get("Section", []):
            yield from _walk_information(sub)

def extract_ontology_terms(view: dict) -> list[str]:
    """
    Devuelve una lista ordenada y sin duplicados de términos de ontología.
    1) Busca sección 'Ontology' (recursiva); si no existe, cae a 'Ontology Summary'.
    """
    secs = view.get("Record", {}).get("Section", [])
    seen, terms = set(), []

    # –– 1) Sección 'Ontology' propiamente dicha ––––––––––––––––––
    onto_sec = find_section(secs, "Ontology")
    if onto_sec:
        for inf in _walk_information(onto_sec):
            for s in _collect_strings(inf.get("Value", {})):
                t = _clean_term(s)
                if t and t not in seen:
                    seen.add(t);  terms.append(t)

    # –– 2) Fallback 'Record Description' → 'Ontology Summary' ––––
    if not terms:
        rec_desc = find_section(secs, "Record Description")
        for inf in rec_desc.get("Information", []):
            if not str(inf.get("Description", "")).lower().startswith("ontology"):
                continue
            for blob in _collect_strings(inf.get("Value", {})):
                # a/​an <algo> ,  an <algo> , …
                for raw in re.findall(r"\b(?:a|an)\s+([^.,;]{1,60})", blob, flags=re.I):
                    if "EC" in raw:           # descarta “EC 3.1.4.*”
                        continue
                    t = _clean_term(raw)
                    if t and t not in seen and 1 <= len(t.split()) <= 5 and re.search("[a-z]", t):
                        seen.add(t);  terms.append(t)

                # “is a trimethylxanthine …”  (captura la clase después de ‘is a’)
                for raw in re.findall(r"\bis a\s+([a-z][^.;]{1,40})", blob, flags=re.I):
                    t = _clean_term(raw)
                    if t and t not in seen and "trimethylxanthine" in t:
                        seen.add(t);  terms.insert(0, t)   # lo ponemos al principio

    return terms

# ---------------------------------------------------
# tu función de etiquetado químico
# ---------------------------------------------------
def _derive_chem_tag(smiles: str, ontology: List[str]) -> str:
    if ontology:
        return ", ".join(ontology[:2])

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "unclassified compound"

    if m.HasSubstructMatch(Chem.MolFromSmarts("P(=O)(O)O")):
        return "organophosphate"
    if m.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1F")):
        return "halogenated aromatic"
    if all(at.GetAtomicNum() in (6, 1) for at in m.GetAtoms()):
        return "hydrocarbon"

    return "unclassified compound"


def build_parsed(
    raw: Dict[str, Any],
    synonyms: Optional[Dict[str, Any]],
    props: Optional[Dict[str, Any]],
    view: Optional[Dict[str, Any]],
    cid: int,
    raw_path: Path,
) -> Dict[str, Any]:
    # --------------------------------------------------
    # 1) Estructura 3D (igual que antes)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2) Propiedades básicas
    # --------------------------------------------------
    p = props.get("PropertyTable", {}).get("Properties", [{}])[0] if props else {}
    smiles = p.get("CanonicalSMILES")
    logp = p.get("XLogP")
    formal_charge = p.get("Charge")

    # --------------------------------------------------
    # 3) Solubilidad (RDKit)
    # --------------------------------------------------
    logs = esol_logS(smiles) if RDKit_OK else None
    pka_vals = None  # pKa predicción deshabilitada

    # --------------------------------------------------
    # 4) Helper PUG‑View
    # --------------------------------------------------
    view_secs = view.get("Record", {}).get("Section", []) if view else []
    def info_fn(heading: str) -> List[Dict[str, Any]]:
        return find_section(view_secs, heading).get("Information", [])

    # 4‑a) Termodinámica
    tinfo = info_fn("Thermodynamics")
    delta_h       = _extract_number(tinfo, "Standard Enthalpy of Formation")
    entropy       = _extract_number(tinfo, "Standard Molar Entropy")
    heat_capacity = _extract_number(tinfo, "Heat Capacity")

    # 4‑b) Seguridad (H‑codes)
    ghs_codes = extract_h_codes(view_secs)

    flash = _extract_number(info_fn("Physical Properties"), "Flash Point")
    ld50  = _extract_number(info_fn("Toxicity"), "LD50")

    # 4‑c) Espectros (sin cambios)
    spec         = find_section(view_secs, "Spectral Information").get("Section", [])
    spectra_raw  = {sub.get("TOCHeading"): sub.get("Information", []) for sub in spec}

    # --------------------------------------------------
    # 5) Sinónimos
    # --------------------------------------------------
    syns       = synonyms.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", []) if synonyms else []
    preferred  = syns[0] if syns else None
    cas_like   = next((s for s in syns if re.fullmatch(r"\d+-\d+-\d+", s)), None)

    # --------------------------------------------------
    # 6) Huellas moleculares
    # --------------------------------------------------
    ecfp = maccs = None
    if RDKit_OK and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            try:
                from rdkit.Chem.AllChem import MorganGenerator
                mg   = MorganGenerator(radius=2)
                ecfp = mg.GetFingerprint(mol).ToBitString()
            except Exception:
                ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
            maccs = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        except Exception as e:
            logging.warning("Fingerprint error: %s", e)

    # --------------------------------------------------
    # 7) Ontología + etiqueta química
    # --------------------------------------------------
    ontology_terms = extract_ontology_terms(view) if view else []
    chem_tag       = _derive_chem_tag(smiles, ontology_terms)

    # --------------------------------------------------
    # 8) Ensamblaje final
    # --------------------------------------------------
    return {
        "structure": {
            "xyz": xyz,
            "atom_symbols": atom_symbols,
            "bond_orders": bond_orders,
            "formal_charge": formal_charge,
            "spin_multiplicity": None,
        },
        "quantum": {k: None for k in (
            "h_core", "g_two", "mo_energies", "homo_index",
            "mulliken_charges", "esp_charges",
            "dipole_moment", "quadrupole_moment"
        )},
        "thermo": {
            "standard_enthalpy": delta_h,
            "entropy": entropy,
            "heat_capacity": heat_capacity,
        },
        "safety": {
            "ghs_codes": ghs_codes,
            "flash_point": flash,
            "ld50": ld50,
        },
        "spectra": {
            "raw": spectra_raw,
        },
        "solubility": {
            "logp": logp,
            "logs": logs,
            "pka": pka_vals,
        },
        "search": {
            "cid": cid,
            "inchi": p.get("InChI"),
            "inchikey": p.get("InChIKey"),
            "smiles": smiles,
            "ecfp": ecfp,
            "maccs": maccs,
            "embeddings": {"summary": None, "structure": None},
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
            "source_version": raw.get("Record", {}).get("RecordMetadata", {}).get("ReleaseDate"),
            "cache_path": str(raw_path),
            "ontology": ontology_terms,
            "chem_tag": chem_tag,
        },
    }
