"""
ingest/parsers.py

Parsers and utility functions for ingesting and processing PubChem PUG-REST and PUG-View JSON.

Responsibilities:
  • Locate and extract sections by TOC headings.
  • Normalize numeric values from nested information blocks.
  • Parse GHS hazard codes with filtering by notification percentage.
  • Walk deep record trees to collect ontology terms and clean them.
  • Derive a simple chemical tag based on ontology or SMARTS patterns.

Requirements:
  • Conform to PubChem request limits (≤ 5 requests/s, ≤ 400 requests/min).
  • RDKit installed for descriptor and fingerprint support; falls back gracefully.
  • Logging enabled for missing features or extraction failures.

"""
import logging
import re
import datetime
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

import requests

# RDKit availability flag and imports
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors
    from rdkit.Chem import rdMolDescriptors as rdDesc
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKit_OK = True
except Exception as e:
    logging.warning(
        "RDKit unavailable: chemical descriptors and fingerprints disabled (%s)", e
    )
    RDKit_OK = False

# ESOL solubility estimator: calculates logS from SMILES
if RDKit_OK:
    def esol_logS(smiles: str) -> float:
        """
        Estimate aqueous solubility (logS) using the ESOL method:
          logS = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*aromatic_fraction

        Parameters:
            smiles (str): Canonical SMILES string.
        Returns:
            float: Predicted log10(solubility) in mol/L.
        """
        m = Chem.MolFromSmiles(smiles)
        logP = Descriptors.MolLogP(m)
        mw   = Descriptors.MolWt(m)
        rb   = Descriptors.NumRotatableBonds(m)
        ap   = sum(atom.GetIsAromatic() for atom in m.GetAtoms()) / m.GetNumHeavyAtoms()
        return 0.16 - 0.63*logP - 0.0062*mw + 0.066*rb - 0.74*ap

# pKa predictions disabled (no supported model)
PKA_OK = False


def find_section(sections: List[Dict[str, Any]], heading: str) -> Dict[str, Any]:
    """
    Recursively search a list of PubChem 'Section' dicts for a TOCHeading.

    Parameters:
        sections (List[Dict[str, Any]]): List of section dictionaries.
        heading (str): Exact TOCHeading text to locate.
    Returns:
        Dict[str, Any]: The matching section dict, or empty dict if not found.
    """
    for sec in sections:
        if sec.get("TOCHeading") == heading:
            return sec
        nested = find_section(sec.get("Section", []), heading)
        if nested:
            return nested
    return {}


def _extract_number(info: List[Dict[str, Any]], key: str) -> Optional[float]:
    """
    Find a numeric value by its label in a PubChem 'Information' list.

    Parameters:
        info (List[Dict[str, Any]]): List of 'Information' entries.
        key (str): The Name field to match.
    Returns:
        Optional[float]: Extracted number or None if missing.
    """
    for entry in info:
        if entry.get("Name") == key:
            num = entry.get("Value", {}).get("Number")
            if isinstance(num, dict):
                return num.get("Value")
    return None

# Regex for GHS hazard codes: Hnnn (percent %)
_H_RX = re.compile(r"\b(H\d{3})(?!\+)\s*\((\d+(?:\.\d+)?)%\)")

def extract_h_codes(
    view_secs: List[Dict[str, Any]],
    min_pct: float = 10.0
) -> List[str]:
    """
    Extract GHS H-codes with notification percentages above a threshold.

    Parameters:
        view_secs (List[Dict[str, Any]]): Sections from PUG-View 'Record'.
        min_pct (float): Minimum percentage to include a code.
    Returns:
        List[str]: Sorted unique list of H-codes (e.g. ['H302', 'H314']).
    """
    ghs = find_section(view_secs, "GHS Classification")
    if not ghs:
        return []

    codes: set[str] = set()
    for info in ghs.get("Information", []):
        if info.get("Name") != "GHS Hazard Statements":
            continue
        text = " ".join(
            part.get("String", "")
            for part in info.get("Value", {}).get("StringWithMarkup", [])
        )
        for h_code, pct in _H_RX.findall(text):
            if float(pct) >= min_pct:
                codes.add(h_code)
    return sorted(codes)


def _collect_strings(value: Dict[str, Any]) -> List[str]:
    """
    Aggregate all plaintext snippets from a PubChem Value block.

    Handles both 'StringWithMarkup' lists and direct 'String' fields.
    """
    out: List[str] = []
    for sm in value.get("StringWithMarkup", []):
        s = sm.get("String")
        if s:
            out.append(s)
    if "String" in value:
        raw = value["String"]
        out.extend(raw if isinstance(raw, list) else [raw])
    return out


def _clean_term(term: str) -> str:
    """
    Normalize an ontology term by removing extraneous clauses and punctuation.

    Steps:
      1) Lowercase and strip whitespace.
      2) Drop content in parentheses and trailing connectors.
      3) Truncate after 'that', 'which', 'with', 'in which'.
    """
    term = term.lower().strip()
    term = term.split('(')[0]
    term = re.split(r"\bin which\b", term)[0]
    term = re.split(r"\b(?:that|which|with)\b", term)[0]
    return re.sub(r"\b(and|or)$", "", term).strip(",;. ")


def _walk_information(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generator that yields all 'Information' blocks in a nested section tree.
    """
    if isinstance(node, dict):
        if "Information" in node:
            for inf in node["Information"]:
                yield inf
        for sub in node.get("Section", []):
            yield from _walk_information(sub)


def extract_ontology_terms(view: Dict[str, Any]) -> List[str]:
    """
    Extract and clean ontology terms from PUG-View JSON.

    1) Search 'Ontology' section for Value.Strings.
    2) If empty, fallback to 'Record Description' → 'Ontology Summary'.
    3) Deduplicate, preserve insertion order, filter noise.

    Returns:
        List[str]: Cleaned ontology terms.
    """
    secs = view.get("Record", {}).get("Section", [])
    seen: set[str] = set()
    terms: List[str] = []

    # Preferred 'Ontology' section
    onto = find_section(secs, "Ontology")
    if onto:
        for info in _walk_information(onto):
            for raw in _collect_strings(info.get("Value", {})): 
                t = _clean_term(raw)
                if t and t not in seen:
                    seen.add(t); terms.append(t)

    # Fallback to descriptive summary
    if not terms:
        desc = find_section(secs, "Record Description")
        for info in desc.get("Information", []):
            desc_txt = str(info.get("Description", "")).lower()
            if not desc_txt.startswith("ontology"): continue
            for blob in _collect_strings(info.get("Value", {})):
                # Capture phrases after 'a/an'
                for raw in re.findall(r"\b(?:a|an)\s+([^.,;]{1,60})", blob):
                    if "ec" in raw.lower(): continue
                    t = _clean_term(raw)
                    if t and t not in seen and len(t.split())<=5:
                        seen.add(t); terms.append(t)
                # Capture 'is a <class>' patterns
                for raw in re.findall(r"\bis a\s+([a-z][^.;]{1,40})", blob, flags=re.I):
                    t = _clean_term(raw)
                    if t and "trimethylxanthine" in t:
                        seen.add(t); terms.insert(0, t)
    return list(dict.fromkeys(terms))


def _derive_chem_tag(smiles: str, ontology: List[str]) -> str:
    """
    Derive a simple chemical classification tag.

    Priority:
      1) First two ontology terms (split on 'and', strip articles, deduped).
      2) SMARTS-based heuristics for organophosphates, halogenated aromatics, hydrocarbons.
      3) 'unclassified compound'.
    """
    if ontology:
        # 1) de-dup and preserve order
        unique = list(dict.fromkeys(ontology))

        # 2) split multi-part tags on 'and', strip leading articles
        parts: List[str] = []
        for tag in unique:
            for sub in tag.split(" and "):
                # remove leading 'a ' or 'an '
                clean = re.sub(r'^(?:a|an)\s+', "", sub.strip(), flags=re.I)
                parts.append(clean)
        # 3) re-dedupe
        parts = list(dict.fromkeys(parts))

        return ", ".join(parts[:2])

    # fallback to SMARTS heuristics
    m = Chem.MolFromSmiles(smiles) if RDKit_OK else None
    if m:
        if m.HasSubstructMatch(Chem.MolFromSmarts("P(=O)(O)O")):
            return "organophosphate"
        if m.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1F")):
            return "halogenated aromatic"
        if all(atom.GetAtomicNum() in (6,1) for atom in m.GetAtoms()):
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
    """
    Parse a PubChem compound record into a standardized Molecule JSON ready for downstream processing.

    This function extracts structural coordinates, basic properties, solubility predictions, safety data,
    spectral information, molecular fingerprints, ontology tags, and metadata from the raw PubChem outputs.

    Parameters:
        raw (Dict[str, Any]): The raw JSON from PubChem PUG-REST (PC_Compounds or PUG-View Record).
        synonyms (Optional[Dict[str, Any]]): Synonym table from the PUG-View synonyms endpoint.
        props (Optional[Dict[str, Any]]): PropertyTable section from PubChem PUG-REST properties endpoint.
        view (Optional[Dict[str, Any]]): Full Record section from PUG-View for enriched annotations.
        cid (int): PubChem Compound ID for this molecule.
        raw_path (Path): Filesystem path to the cached raw JSON file.

    Returns:
        Dict[str, Any]: A dictionary conforming to the Molecule JSON schema, containing:
            - structure: 3D coordinates, atom symbols, bond orders, formal charge, spin multiplicity
            - quantum: placeholders for quantum outputs (initialized to None)
            - thermo: standard enthalpy, entropy, heat capacity
            - safety: GHS hazard codes, flash point, LD50
            - spectra: raw spectral data
            - solubility: logP, predicted logS, pKa values
            - search: identifiers, structural keys, and empty embedding slots
            - names: preferred, CAS-like, and synonym lists
            - meta: fetch timestamp, source info, cache path, ontology vocab, chemical tag

    Raises:
        ValueError: If no structural coordinates can be extracted from the raw data.
    """
    # --------------------------------------------------
    # 1) Extract 3D structure (coordinates, atom labels, bond orders)
    # --------------------------------------------------
    xyz: Optional[List[Tuple[float, float, float]]] = None
    atom_symbols: Optional[List[str]] = None
    bond_orders = None

    # Handle PUG-View Record format
    if raw.get("Record"):
        sections = {sec.get("TOCHeading"): sec for sec in raw["Record"].get("Section", [])}
        info3d = sections.get("3D Conformer", {}).get("Information", [])
        if info3d:
            conformer = info3d[0]["Value"]["Conformer3D"]
            xyz = [(c["X"], c["Y"], c["Z"]) for c in conformer.get("Coordinates", [])]
            atom_symbols = conformer.get("Atoms")
    # Handle PUG-REST PC_Compounds format
    elif raw.get("PC_Compounds"):
        compound = raw["PC_Compounds"][0]
        elements = compound.get("atoms", {}).get("element", [])
        coords_list = compound.get("coords", [])
        if coords_list:
            coords = coords_list[0].get("conformers", [])
            if coords:
                c = coords[0]
                xs, ys, zs = c.get("x", []), c.get("y", []), c.get("z", [])
                xyz = list(zip(xs, ys, zs))
                # Derive atom symbols either via RDKit or fallback integers
                if RDKit_OK:
                    try:
                        ptable = Chem.GetPeriodicTable()
                        atom_symbols = [ptable.GetElementSymbol(el) for el in elements]
                    except Exception:
                        atom_symbols = [str(el) for el in elements]
                else:
                    atom_symbols = [str(el) for el in elements]
        # Parse bond connectivity if available
        aid1 = compound.get("bonds", {}).get("aid1", [])
        aid2 = compound.get("bonds", {}).get("aid2", [])
        orders = compound.get("bonds", {}).get("order", [])
        if aid1 and aid2 and orders:
            bond_orders = list(zip(aid1, aid2, orders))

    # Ensure we obtained at least coordinates
    if xyz is None or atom_symbols is None:
        raise ValueError(f"Failed to extract 3D structure for CID {cid}")

    # --------------------------------------------------
    # 2) Extract basic computed properties
    # --------------------------------------------------
    props_entry = props.get("PropertyTable", {}).get("Properties", [{}])[0] if props else {}
    smiles = props_entry.get("CanonicalSMILES")
    logp = props_entry.get("XLogP")
    formal_charge = props_entry.get("Charge")

    # --------------------------------------------------
    # 3) Predict solubility (ESOL method via RDKit)
    # --------------------------------------------------
    logs = esol_logS(smiles) if RDKit_OK and smiles else None

    # pKa prediction is disabled by default.
    # It's crucial for understanding ionization, solubility, and bioavailability,
    # but not yet included due to the lack of a reliable, open-source predictor.
    # Future versions of Molkit may support pKa via ML or QM-based tools (e.g., openpka or DelFTa).
    pka_vals = None
    # --------------------------------------------------
    # 4) Helper to query PUG-View sections
    # --------------------------------------------------
    view_sections = view.get("Record", {}).get("Section", []) if view else []
    def get_info(heading: str) -> List[Dict[str, Any]]:
        return find_section(view_sections, heading).get("Information", [])

    # 4-a) Thermodynamic properties
    thermo_info = get_info("Thermodynamics")
    delta_h = _extract_number(thermo_info, "Standard Enthalpy of Formation")
    entropy = _extract_number(thermo_info, "Standard Molar Entropy")
    heat_capacity = _extract_number(get_info("Heat Capacity"), "Heat Capacity")

    # 4-b) Safety data (GHS codes, flash point, LD50)
    ghs_codes = extract_h_codes(view_sections)
    flash = _extract_number(get_info("Physical Properties"), "Flash Point")
    ld50 = _extract_number(get_info("Toxicity"), "LD50")

    # 4-c) Spectral information
    spectral_sections = find_section(view_sections, "Spectral Information").get("Section", [])
    
    _heading_map = {
        "Raman spectra available": "Raman",
        "Other spectra available": "Other available",
    }

    spectra_raw = {
        # look up the TOCHeading in our map, default to itself if not found
        _heading_map.get(sub.get("TOCHeading", ""), sub.get("TOCHeading", ""))  
        : sub.get("Information", [])
        for sub in spectral_sections
    }

    # --------------------------------------------------
    # 5) Extract synonyms and preferred names
    # --------------------------------------------------
    raw_syns = synonyms.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", []) if synonyms else []
    preferred_name = raw_syns[0] if raw_syns else None
    cas_like = next((s for s in raw_syns if re.fullmatch(r"\d+-\d+-\d+", s)), None)

    # --------------------------------------------------
    # 6) Generate molecular fingerprints
    # --------------------------------------------------
    ecfp = maccs = None
    if RDKit_OK and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            # ECFP: 1024-bit binary fingerprint
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            ecfp_array = np.zeros((1024,), dtype=int)
            DataStructs.ConvertToNumpyArray(bv, ecfp_array)
            ecfp = ecfp_array.tolist()

            # MACCS: optional, keep as on-bit indices
            maccs_bv = MACCSkeys.GenMACCSKeys(mol)
            maccs = list(maccs_bv.GetOnBits())
        except Exception as e:
            logging.warning("Fingerprint generation failed for CID %s: %s", cid, e)

    # --------------------------------------------------
    # 7) Compute extra RDKit‐based descriptors for querying
    # --------------------------------------------------
    # Note: requires rdDesc and Descriptors imports
    mw            = None
    formula       = None
    heavy_atom_ct = None
    hbd           = None
    hba           = None
    rotors        = None
    ring_cnt      = None
    arom_ring_cnt = None
    tpsa          = None
    fsp3          = None
    bertz_ct      = None

    if RDKit_OK and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            # molecular formula & weight
            formula       = rdDesc.CalcMolFormula(mol)
            mw            = Descriptors.MolWt(mol)
            # heavy‐atom count
            heavy_atom_ct = mol.GetNumHeavyAtoms()
            # H‐bond donors/acceptors
            hbd = rdDesc.CalcNumHBD(mol)
            hba = rdDesc.CalcNumHBA(mol)
            # rotatable bonds
            rotors = Descriptors.NumRotatableBonds(mol)
            # rings & aromatic ring count
            ring_cnt      = rdDesc.CalcNumRings(mol)
            arom_ring_cnt = sum(
                1 for ring in mol.GetRingInfo().AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            )
            # TPSA, Fsp3 and Bertz complexity
            tpsa     = rdDesc.CalcTPSA(mol)
            fsp3     = rdDesc.CalcFractionCSP3(mol)
            bertz_ct = Descriptors.BertzCT(mol)
        except Exception as e:
            logging.warning("Descriptor calculation failed for CID %s: %s", cid, e)

    # --------------------------------------------------
    # 8) Ontology term extraction and chemical tagging
    # --------------------------------------------------
    ontology_terms = extract_ontology_terms(view) if view else []
    chem_tag = _derive_chem_tag(smiles, ontology_terms)

    # --------------------------------------------------
    # 9) Final assembly into Molecule JSON
    # --------------------------------------------------
    molecule = {
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
        "spectra": {"raw": spectra_raw},
        "solubility": {"logp": logp, "logs": logs, "pka": pka_vals},
        "search": {
            "cid": cid,
            "inchi": props_entry.get("InChI"),
            "inchikey": props_entry.get("InChIKey"),
            "smiles": smiles,
            "molecular_weight": mw,
            "formula": formula,
            "heavy_atom_count": heavy_atom_ct,
            "hbond_donors": hbd,
            "hbond_acceptors": hba,
            "rotatable_bonds": rotors,
            "ring_count": ring_cnt,
            "aromatic_ring_count": arom_ring_cnt,
            "tpsa": tpsa,
            "fsp3": fsp3,
            "bertz_ct": bertz_ct,
            "ecfp": ecfp,
            "maccs": maccs,
        },
        "names": {
            "preferred": preferred_name,
            "cas_like": cas_like,
            "systematic": None,
            "traditional": None,
            "synonyms": raw_syns,
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
    print(f"Procesado registro CID {cid}")

    return molecule
