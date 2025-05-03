from robotu_molkit.vector.summary_generator import SummaryGenerator
import json, pathlib

data = json.loads(pathlib.Path("data/parsed/pubchem_2519.json").read_text())
sg = SummaryGenerator()          # credenciales se cargan del ~/.config/molkit
summary = sg.generate_all_summaries(data)
print(summary)
