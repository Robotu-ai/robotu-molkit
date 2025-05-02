import typer
from robotu_molkit.ingest.cli import ingest_app

app = typer.Typer(add_completion=False)

app.add_typer(ingest_app, name="ingest")  # <- este es el subcomando

def main() -> None:
    app()

if __name__ == "__main__":
    main()
