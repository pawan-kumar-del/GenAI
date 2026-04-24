from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')
import typer
from rich.console import Console
from rich.prompt import Prompt

from rag_support_assistant.config import get_config
from rag_support_assistant.logging_utils import configure_logging
from rag_support_assistant.service import run_support_query

app = typer.Typer(help="RAG-based customer support assistant CLI.")
console = Console()


@app.command()
def ask(query: str) -> None:
    """Ask one question and print the workflow result."""
    config = get_config()
    configure_logging(config.log_level)

    try:
        final_state = run_support_query(config=config, user_query=query)
        answer = final_state.get("answer", "No answer returned.")
        confidence = float(final_state.get("confidence", 0.0))
        escalated = bool(final_state.get("needs_escalation", False))

        console.print(f"[bold cyan]Answer:[/bold cyan] {answer}")
        console.print(
            f"[dim]confidence={confidence:.2f}, escalated={str(escalated).lower()}[/dim]"
        )
    except Exception as error:
        console.print(f"[red]Request failed:[/red] {error}")
        raise typer.Exit(code=1) from error


@app.command()
def chat() -> None:
    """Start interactive customer support assistant session."""
    config = get_config()
    configure_logging(config.log_level)
    console.print("[bold green]Customer Support Assistant is ready.[/bold green]")
    console.print("Type 'exit' to quit.")

    while True:
        user_query = Prompt.ask("\n[bold yellow]You[/bold yellow]")
        if user_query.strip().lower() in {"exit", "quit"}:
            console.print("[bold]Session ended.[/bold]")
            break

        try:
            final_state = run_support_query(config=config, user_query=user_query)
            answer = final_state.get("answer", "No answer returned.")
            confidence = float(final_state.get("confidence", 0.0))
            escalated = bool(final_state.get("needs_escalation", False))
            console.print(f"[bold cyan]Assistant:[/bold cyan] {answer}")
            console.print(
                f"[dim]confidence={confidence:.2f}, escalated={str(escalated).lower()}[/dim]"
            )
        except Exception as error:
            console.print(f"[red]Error:[/red] {error}")


if __name__ == "__main__":
    app()

