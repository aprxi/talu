#!/usr/bin/env python3
"""
Batch CSV Processor - High-throughput parallel prompt processing.

Job: Process thousands of prompts from CSV using async concurrency with progress tracking.
Prereqs: rich, aiofiles
Failure mode: Prints "Setup: pip install rich aiofiles" and exits with code 1.

Demonstrates:
- AsyncClient for non-blocking I/O
- Semaphore-based concurrency control
- Progress tracking with rich
- Error handling and retry logic
- Structured output for consistent parsing
- Resume capability for interrupted jobs

Requirements:
    pip install rich aiofiles

Run:
    python examples/recipes/batch_csv_processor.py --generate-sample
    python examples/recipes/batch_csv_processor.py input.csv output.csv
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    import aiofiles
except ImportError:
    print("Setup: pip install rich aiofiles")
    sys.exit(1)

from pydantic import BaseModel

import talu
from talu import AsyncClient
from talu.router import GenerationConfig


# =============================================================================
# Schema for structured output
# =============================================================================

class ProcessedResult(BaseModel):
    """Structured output for each processed prompt."""
    summary: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    keywords: list[str]


# =============================================================================
# Statistics tracking
# =============================================================================

@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_tokens: int = 0
    start_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.completed + self.failed == 0:
            return 0.0
        return self.completed / (self.completed + self.failed) * 100

    @property
    def throughput(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.completed / elapsed


# =============================================================================
# Core processing logic
# =============================================================================

async def process_single(
    client: AsyncClient,
    row_id: str,
    prompt: str,
    config: GenerationConfig,
    semaphore: asyncio.Semaphore,
    stats: ProcessingStats,
    max_retries: int = 3,
) -> tuple[str, Optional[dict], Optional[str]]:
    """
    Process a single prompt with retry logic.

    Returns:
        Tuple of (row_id, result_dict, error_message)
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                chat = client.chat(system="You are an analyst. Analyze the given text.")
                response = await chat.send(
                    f"Analyze this text:\n\n{prompt}",
                    response_format=ProcessedResult,
                    config=config,
                )

                # Extract result
                result = response.parsed
                stats.completed += 1
                if response.usage:
                    stats.total_tokens += response.usage.total_tokens

                return (row_id, result.model_dump(), None)

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    stats.failed += 1
                    return (row_id, None, str(e))

    return (row_id, None, "Unknown error")


async def process_batch(
    input_path: Path,
    output_path: Path,
    model: str,
    concurrency: int,
    max_tokens: int,
    resume: bool = False,
) -> ProcessingStats:
    """Process all rows from input CSV and write results to output CSV."""
    console = Console()

    # Load existing results if resuming
    completed_ids: set[str] = set()
    if resume and output_path.exists():
        async with aiofiles.open(output_path, "r") as f:
            content = await f.read()
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                if row.get("status") == "success":
                    completed_ids.add(row["id"])
        console.print(f"[yellow]Resuming: {len(completed_ids)} already completed[/yellow]")

    # Read input CSV
    rows = []
    async with aiofiles.open(input_path, "r") as f:
        content = await f.read()
        reader = csv.DictReader(content.splitlines())
        for row in reader:
            if row["id"] not in completed_ids:
                rows.append(row)

    if not rows:
        console.print("[green]All rows already processed![/green]")
        return ProcessingStats()

    # Initialize
    stats = ProcessingStats(total=len(rows), skipped=len(completed_ids))
    stats.start_time = time.time()

    config = GenerationConfig(
        max_tokens=max_tokens,
        temperature=0.3,  # Lower temperature for consistency
    )

    semaphore = asyncio.Semaphore(concurrency)

    # Open output file for appending
    write_header = not output_path.exists() or not resume
    output_file = await aiofiles.open(output_path, "a" if resume else "w", newline="")

    try:
        # Write header if needed
        if write_header:
            header = "id,prompt,status,summary,sentiment,confidence,keywords,error\n"
            await output_file.write(header)

        async with AsyncClient(model) as client:
            # Create tasks
            tasks = [
                process_single(
                    client=client,
                    row_id=row["id"],
                    prompt=row["prompt"],
                    config=config,
                    semaphore=semaphore,
                    stats=stats,
                )
                for row in rows
            ]

            # Process with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Processing {len(rows)} prompts...",
                    total=len(rows),
                )

                # Process results as they complete
                for coro in asyncio.as_completed(tasks):
                    row_id, result, error = await coro

                    # Find original prompt
                    original_row = next(r for r in rows if r["id"] == row_id)
                    prompt = original_row["prompt"]

                    # Write result row
                    if result:
                        line = (
                            f'"{row_id}","{prompt[:50]}...","success",'
                            f'"{result["summary"]}","{result["sentiment"]}",'
                            f'{result["confidence"]},"{",".join(result["keywords"])}",\n'
                        )
                    else:
                        line = f'"{row_id}","{prompt[:50]}...","failed",,,,,,"{error}"\n'

                    await output_file.write(line)
                    await output_file.flush()

                    progress.update(task, advance=1)

    finally:
        await output_file.close()

    return stats


def generate_sample_csv(path: Path, num_rows: int = 100):
    """Generate a sample input CSV for testing."""
    console = Console()

    sample_texts = [
        "I absolutely love this product! It exceeded all my expectations.",
        "The service was terrible and I want a refund immediately.",
        "It's okay, nothing special but gets the job done.",
        "Best purchase I've made this year. Highly recommend!",
        "Disappointed with the quality. Not worth the price.",
        "Amazing customer support, they resolved my issue quickly.",
        "The delivery was late and the package was damaged.",
        "Exactly what I needed. Perfect for my use case.",
        "Could be better. Some features are missing.",
        "Five stars! Will definitely buy again.",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt"])
        for i in range(num_rows):
            text = sample_texts[i % len(sample_texts)]
            # Add some variation
            if i % 3 == 0:
                text = f"Review #{i}: {text}"
            elif i % 3 == 1:
                text = f"Customer feedback: {text} (Order #{i})"
            else:
                text = f"{text} - Submitted by user_{i}"
            writer.writerow([f"row_{i:04d}", text])

    console.print(f"[green]Generated sample CSV: {path} ({num_rows} rows)[/green]")


def print_summary(stats: ProcessingStats, console: Console):
    """Print processing summary."""
    elapsed = time.time() - stats.start_time

    table = Table(title="Processing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total rows", str(stats.total))
    table.add_row("Completed", str(stats.completed))
    table.add_row("Failed", str(stats.failed))
    table.add_row("Skipped (resumed)", str(stats.skipped))
    table.add_row("Success rate", f"{stats.success_rate:.1f}%")
    table.add_row("Total tokens", f"{stats.total_tokens:,}")
    table.add_row("Elapsed time", f"{elapsed:.1f}s")
    table.add_row("Throughput", f"{stats.throughput:.2f} rows/sec")

    console.print(table)


async def main():
    parser = argparse.ArgumentParser(
        description="Batch CSV processor with async concurrency"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV file path",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen3-0.6B",
        help="Model to use (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per response (default: 256)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed rows)",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a sample input CSV",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100,
        help="Number of rows in sample CSV (default: 100)",
    )
    args = parser.parse_args()

    console = Console()

    # Generate sample mode
    if args.generate_sample:
        sample_path = Path(args.input or "sample_input.csv")
        generate_sample_csv(sample_path, args.sample_rows)
        return

    # Validate args
    if not args.input or not args.output:
        parser.error("Both input and output paths are required")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        console.print("[dim]Run with --generate-sample to create a test file[/dim]")
        return

    # Print config
    console.print(Panel(
        f"[bold]Batch CSV Processor[/bold]\n\n"
        f"Input:       [cyan]{input_path}[/cyan]\n"
        f"Output:      [cyan]{output_path}[/cyan]\n"
        f"Model:       [cyan]{args.model}[/cyan]\n"
        f"Concurrency: [cyan]{args.concurrency}[/cyan]\n"
        f"Resume:      [cyan]{args.resume}[/cyan]",
        border_style="blue",
    ))

    # Process
    stats = await process_batch(
        input_path=input_path,
        output_path=output_path,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        resume=args.resume,
    )

    # Print summary
    if stats.total > 0:
        print_summary(stats, console)
        console.print(f"\n[green]Results written to: {output_path}[/green]")


if __name__ == "__main__":
    asyncio.run(main())

"""
Topics covered:
* batch.processing
* structured.output
* workflow.end.to.end
"""
