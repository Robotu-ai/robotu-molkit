# 🧪 robotu-molkit

> 🚧 **This project is under active development.** Expect frequent changes as we build the foundation for quantum-ready molecular discovery.

**Quantum-ready molecular toolkit.**  
An open-source Python library for retrieving, structuring, and enriching molecular data from PubChem — ready for quantum simulation, AI pipelines, and scientific research.

---

## 🔍 About

**robotu-molkit** is part of the **RobotU Quantum** ecosystem — an open-source Python library that automates the extraction, structuring, and enrichment of chemical data directly from **PubChem’s XML-based APIs**.

The library retrieves and organizes a wide range of chemical property categories — including geometric, thermodynamic, quantum, and spectroscopic data — into a unified data model, making it instantly usable for quantum frameworks like **Qiskit**, AI pipelines, and molecular modeling tools.

---

## ✅ Features

- ⚛️ **Quantum-ready**: Seamless integration with Qiskit and other quantum simulation frameworks  
- 🔗 **Direct PubChem API access**: Connects to PubChem’s XML APIs — no scraping, no manual downloads  
- 🧠 **AI-powered parsing and structuring**: Uses IBM Granite to parse complex XML responses and organize molecular data into structured Python objects  
- 🧬 **10+ property categories**: Automatically extracts and formats data like geometries, thermodynamics, quantum/electronic properties, spectroscopy, and more  
- 🔁 **Batch-friendly**: Easily iterate through thousands of molecules programmatically  
- 📦 **Cloud-free**: Entirely open-source and locally executable — no dependency on external cloud infrastructure

---

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


