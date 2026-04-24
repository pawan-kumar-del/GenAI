from __future__ import annotations

from pathlib import Path
import sys
import typer
from rich.console import Console

from rag_support_assistant.config import get_config
from rag_support_assistant.logging_utils import configure_logging
from rag_support_assistant.service import ingest_knowledge_base

app = typer.Typer(help="Ingest a PDF knowledge base into ChromaDB.")
console = Console()


@app.command()
def run(pdf_path: str | None = typer.Option(None, help="Path to PDF knowledge base.")) -> None:
    """Ingest support documentation PDF into the vector database."""
    config = get_config()
    configure_logging(config.log_level)
    knowledge_base_path = config.default_pdf_path if pdf_path is None else Path(pdf_path)

    try:
        chunk_count = ingest_knowledge_base(config, pdf_path=knowledge_base_path)
        console.print(
            f"[green]Ingestion complete.[/green] Stored [bold]{chunk_count}[/bold] chunks."
        )
    except Exception as error:
        console.print(f"[red]Ingestion failed:[/red] {error}")
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.argv.pop(1)
    app()

