# 🧪 robotu-molkit

> 🚧 **This project is under active development.** Expect frequent changes as we build the foundation for quantum-ready molecular discovery.

**Quantum-ready molecular toolkit.**  
An open-source Python library for retrieving, structuring, and enriching molecular data from PubChem — ready for quantum simulation, AI pipelines, and scientific research.

---

## 🔍 About

**robotu-molkit** is part of the **RobotU Quantum** ecosystem.  
Its mission is straightforward:

> **Turn PubChem molecular records into a quantum-ready dataset—complete with AI-generated summaries and vector embeddings.**

The library fetches molecular data directly from PubChem, extracts more than ten property categories (geometry, thermodynamics, electronic, spectroscopic, safety, …), and converts each record into a clean `Molecule` object that drops straight into **Qiskit**, classical MD, or any AI pipeline.  
IBM Granite models enrich every molecule with human-readable metadata and high-dimensional embeddings, enabling semantic similarity search out of the box.

---

## ✅ Key Features

| Capability | What it does |
|------------|--------------|
| ⚛️ **Quantum-ready exporters** | `mol.to_qiskit(basis="sto3g")` (and similar) generate qubit-mappable data for VQE or CCSD workflows. |
| 🔗 **Seamless PubChem access** | Pulls molecular records via official APIs or PubChemPy—no manual downloads required. |
| 🧠 **Granite-powered enrichment** | Granite produces structured Python objects, natural-language summaries, and embeddings for each molecule. |
| 🧬 **Rich property coverage** | >10 property blocks automatically normalized: geometries, thermodynamics, quantum/electronic properties, spectroscopy, solubility, safety, etc. |
| 🔍 **Vector similarity search** | Built-in FAISS index lets you query *“low-toxicity aromatic amines”* or *“molecules similar to caffeine”*. |
| 🔁 **Batch-friendly helper** | `dataset.build_from_query("alkaloid", limit=5000)` fetches and caches large sets with one line. |
| 📦 **Cloud-free & open-source** | Runs locally; no vendor lock-in. Just `pip install robotu-molkit` and you’re ready. |

---

*RobotU Molkit turns molecular data into simulation-ready fuel—so researchers focus on discovery, not data wrangling.*

## 🧪 Preliminary Example

```python
from robotu_molkit import molecule

# Retrieve comprehensive molecular data for caffeine
mol = molecule.get("caffeine")

# Extract structured properties
xyz = mol.to_xyz()
thermo = mol.thermodynamic_properties
quantum = mol.quantum_properties

# Iterate over a list of molecules
for compound in ['caffeine', 'aspirin', 'benzene']:
    data = molecule.get(compound)
    xyz = data.to_xyz()
    # Run simulations or analysis...
```

---

## 🔬 Ideal For

- Quantum chemists and simulation developers  
- AI researchers in drug discovery and material science  
- Educators and students in quantum chemistry or computational science  
- Open science and reproducibility advocates

---

## 📦 Installation

> **Coming soon:** `pip install robotu-molkit`

For now, clone the repo:
```bash
git clone https://github.com/robotu-ai/robotu-molkit.git
cd robotu-molkit
pip install -e .
```

---

## 📄 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](./LICENSE) file for details.

---

## 🧠 Powered By

- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) – public chemical compound data  
- [IBM Granite](https://www.ibm.com/granite) – used for AI-powered data parsing, structuring, and metadata enrichment   
- [Qiskit](https://qiskit.org/) – quantum simulation framework

---

## 🤝 Contribute

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) (coming soon) or open an issue to suggest improvements or new features.

---

## 🌐 Contact

Questions? Feedback? Collaborations?  
📧 tech@robotu.ai  
🔬 [robotu.ai](https://robotu.ai) (coming soon)

---

**RobotU Quantum — accelerating discovery through open, AI-enhanced, quantum-ready data.**


