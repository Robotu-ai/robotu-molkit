import typer
import json
from pathlib import Path
from robotu_molkit.ingest.cli import ingest_app

app = typer.Typer(add_completion=False)

app.add_typer(ingest_app, name="ingest") 

CONFIG_PATH = Path.home() / ".config" / "molkit" / "config.json"

@app.command("config")
def config_set(
    api_key: str = typer.Option(..., "--api-key", "-k", help="IBM API Key"),
    project_id: str = typer.Option(..., "--project-id", "-j", help="IBM Project ID"),
):
    """
    Guarda las credenciales IBM en ~/.config/molkit/config.json
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "api_key": api_key,
        "project_id": project_id
    }, indent=2))
    typer.secho(f"Credenciales guardadas en {CONFIG_PATH}", fg=typer.colors.GREEN)


def main() -> None:
    app()

if __name__ == "__main__":
    main()
