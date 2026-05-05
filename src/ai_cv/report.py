"""Rich terminal output for scoring results."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_cv.models import CVScore


console = Console()


def print_score(result: CVScore) -> None:
    """Print a single CV score result to the terminal."""
    # Header
    grade = _grade(result.final_score)
    color = _grade_color(grade)
    console.print(
        Panel(
            f"[bold]{result.candidate_name}[/bold]\n"
            f"File: {result.file}\n"
            f"Profession: {result.profession}\n"
            f"Experience Level: {result.experience_level}",
            title=f"[{color}]{grade} — {result.final_score:.1f}/100[/{color}]",
            border_style=color,
        )
    )

    # Category breakdown table
    table = Table(title="Score Breakdown", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Skills Detail", max_width=60)

    for cat in result.categories:
        pct = (cat.score / cat.max_points * 100) if cat.max_points > 0 else 0
        cat_color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"

        skills_detail = "\n".join(
            f"  {s.skill}: {s.score}/{s.max_points}"
            + (f" — {s.evidence[:80]}" if s.evidence else "")
            for s in cat.skills
        )

        table.add_row(
            cat.category,
            f"[{cat_color}]{cat.score:.1f}[/{cat_color}]",
            f"{cat.max_points:.0f}",
            skills_detail,
        )

    console.print(table)

    # Red flags
    if result.red_flags:
        console.print("\n[bold red]Red Flags:[/bold red]")
        for rf in result.red_flags:
            console.print(f"  [red]- {rf.flag} ({rf.penalty:+.0f})[/red]: {rf.reason}")
        console.print(
            f"\n  Penalty total: [red]{result.penalty_total:+.1f}[/red]"
        )

    # Summary
    if result.summary:
        console.print(f"\n[bold]Summary:[/bold] {result.summary}")

    console.print()


def print_batch_summary(results: list[CVScore]) -> None:
    """Print a ranked summary table of all scored CVs."""
    ranked = sorted(results, key=lambda r: r.final_score, reverse=True)

    table = Table(title="Batch Results — Ranked", show_lines=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Candidate", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Grade")
    table.add_column("Level")
    table.add_column("Flags", justify="right")

    for i, r in enumerate(ranked, 1):
        grade = _grade(r.final_score)
        color = _grade_color(grade)
        table.add_row(
            str(i),
            r.candidate_name,
            f"[{color}]{r.final_score:.1f}[/{color}]",
            f"[{color}]{grade}[/{color}]",
            r.experience_level,
            str(len(r.red_flags)) if r.red_flags else "0",
        )

    console.print(table)


def export_json(results: list[CVScore], output_path: str | Path) -> None:
    """Export results to a JSON file."""
    output_path = Path(output_path)
    data = [r.model_dump() for r in results]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Results exported to {output_path}[/green]")


def _grade(score: float) -> str:
    if score >= 85:
        return "Exceptional"
    elif score >= 70:
        return "Strong"
    elif score >= 55:
        return "Competent"
    elif score >= 40:
        return "Developing"
    elif score >= 25:
        return "Weak"
    else:
        return "Poor"


def _grade_color(grade: str) -> str:
    return {
        "Exceptional": "bright_green",
        "Strong": "green",
        "Competent": "yellow",
        "Developing": "dark_orange",
        "Weak": "red",
        "Poor": "bright_red",
    }.get(grade, "white")
