"""CLI entry point for the AI CV scoring system."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ai_cv.llm import LLMClient
from ai_cv.profile_loader import list_profiles, load_profile
from ai_cv.report import console, export_json, print_batch_summary, print_score
from ai_cv.scorer import CVScorer

app = typer.Typer(
    name="ai-cv",
    help="AI-powered CV/resume scoring with profession-specific rubrics.",
    no_args_is_help=True,
)


@app.command()
def score(
    cv: Path = typer.Argument(..., help="Path to a CV file (PDF, DOCX, or TXT)"),
    profile: Path = typer.Option(
        "profiles/embedded_engineer.yaml",
        "--profile", "-p",
        help="Path to the profession profile YAML",
    ),
    model: str = typer.Option(
        "gemma3:27b",
        "--model", "-m",
        help="Ollama model name",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "--url",
        help="Ollama server URL",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Export results to JSON file",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature", "-t",
        help="LLM temperature (lower = more consistent)",
    ),
    context_size: int = typer.Option(
        16384,
        "--ctx",
        help="LLM context window size",
    ),
):
    """Score a single CV against a profession profile."""
    if not cv.exists():
        console.print(f"[red]CV file not found: {cv}[/red]")
        raise typer.Exit(1)

    if not profile.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    prof = load_profile(profile)
    llm = LLMClient(
        base_url=base_url,
        model=model,
        temperature=temperature,
        num_ctx=context_size,
    )

    # Health check
    if not llm.check_health():
        console.print(
            "[red]Cannot connect to Ollama.[/red]\n"
            f"Make sure Ollama is running at {base_url}\n"
            f"Run: [bold]ollama serve[/bold] and [bold]ollama pull {model}[/bold]"
        )
        raise typer.Exit(1)

    console.print(f"[bold]Scoring:[/bold] {cv.name}")
    console.print(f"[bold]Profile:[/bold] {prof.profession}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print()

    scorer = CVScorer(prof, llm)

    with console.status("[bold green]Analyzing CV with LLM..."):
        result = scorer.score_cv(cv)

    print_score(result)

    if output:
        export_json([result], output)


@app.command()
def batch(
    cv_dir: Path = typer.Argument(..., help="Directory containing CV files"),
    profile: Path = typer.Option(
        "profiles/embedded_engineer.yaml",
        "--profile", "-p",
        help="Path to the profession profile YAML",
    ),
    model: str = typer.Option(
        "gemma3:27b",
        "--model", "-m",
        help="Ollama model name",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "--url",
        help="Ollama server URL",
    ),
    output: Optional[Path] = typer.Option(
        "output/results.json",
        "--output", "-o",
        help="Export results to JSON file",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature", "-t",
        help="LLM temperature",
    ),
    context_size: int = typer.Option(
        16384,
        "--ctx",
        help="LLM context window size",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Score CVs concurrently (faster but uses more VRAM)",
    ),
):
    """Score all CVs in a directory against a profession profile."""
    if not cv_dir.is_dir():
        console.print(f"[red]Directory not found: {cv_dir}[/red]")
        raise typer.Exit(1)

    # Collect CV files
    extensions = {".pdf", ".docx", ".doc", ".txt"}
    cv_files = sorted(
        f for f in cv_dir.iterdir()
        if f.suffix.lower() in extensions and not f.name.startswith(".")
    )

    if not cv_files:
        console.print(f"[yellow]No CV files found in {cv_dir}[/yellow]")
        raise typer.Exit(0)

    prof = load_profile(profile)
    llm = LLMClient(
        base_url=base_url,
        model=model,
        temperature=temperature,
        num_ctx=context_size,
    )

    if not llm.check_health():
        console.print(
            "[red]Cannot connect to Ollama.[/red]\n"
            f"Make sure Ollama is running at {base_url}\n"
            f"Run: [bold]ollama serve[/bold] and [bold]ollama pull {model}[/bold]"
        )
        raise typer.Exit(1)

    console.print(f"[bold]Batch scoring {len(cv_files)} CVs[/bold]")
    console.print(f"[bold]Profile:[/bold] {prof.profession}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print()

    scorer = CVScorer(prof, llm)

    if parallel:
        results = asyncio.run(scorer.score_batch_async(cv_files))
    else:
        results = []
        for i, cv_file in enumerate(cv_files, 1):
            console.print(f"[{i}/{len(cv_files)}] Scoring {cv_file.name}...")
            try:
                result = scorer.score_cv(cv_file)
                results.append(result)
                console.print(
                    f"  -> {result.candidate_name}: "
                    f"[bold]{result.final_score:.1f}/100[/bold]"
                )
            except Exception as e:
                console.print(f"  -> [red]Error: {e}[/red]")

    # Print summary
    console.print()
    print_batch_summary(results)

    # Detailed results
    for result in sorted(results, key=lambda r: r.final_score, reverse=True):
        print_score(result)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        export_json(results, output)


@app.command()
def profiles():
    """List available profession profiles."""
    found = list_profiles()
    if not found:
        console.print("[yellow]No profiles found in profiles/ directory[/yellow]")
        return

    console.print("[bold]Available profiles:[/bold]")
    for p in found:
        prof = load_profile(p)
        console.print(f"  - {p.name}: [bold]{prof.profession}[/bold] (v{prof.version})")


@app.command()
def check():
    """Check Ollama connectivity and list available models."""
    llm = LLMClient()
    if not llm.check_health():
        console.print("[red]Ollama is not running at http://localhost:11434[/red]")
        console.print("Start it with: [bold]ollama serve[/bold]")
        raise typer.Exit(1)

    console.print("[green]Ollama is running.[/green]")
    models = llm.list_models()
    if models:
        console.print("[bold]Available models:[/bold]")
        for m in models:
            console.print(f"  - {m}")
    else:
        console.print("[yellow]No models installed. Run: ollama pull gemma3:27b[/yellow]")


if __name__ == "__main__":
    app()
