#!/usr/bin/env python3
"""
Interactive Terminal Chat - A sophisticated CLI chatbot with streaming.

Job: Provide an interactive terminal chat with streaming, history, and special commands.
Prereqs: rich, prompt_toolkit
Failure mode: Prints "Setup: pip install rich prompt_toolkit" and exits with code 1.

Features:
- Real-time streaming with rich formatting
- Graceful Ctrl+C handling (interrupt generation, not exit)
- Command history with up/down arrow navigation
- Special commands (/clear, /history, /system, /exit)
- Thinking mode toggle for reasoning models
- Token usage tracking

Requirements:
    pip install rich prompt_toolkit

Run:
    python examples/recipes/cli_streaming_ui.py
    python examples/recipes/cli_streaming_ui.py --model Qwen/Qwen3-0.6B
"""

import argparse
import signal
import sys
from typing import Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
except ImportError:
    print("Setup: pip install rich prompt_toolkit")
    sys.exit(1)

import talu
from talu.router import GenerationConfig


class InteractiveChat:
    """Interactive terminal chat with streaming and rich formatting."""

    def __init__(
        self,
        model: str,
        system: Optional[str] = None,
        thinking: bool = False,
        max_tokens: int = 1024,
    ):
        self.console = Console()
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens

        # Create chat session
        config = GenerationConfig(
            max_tokens=max_tokens,
            allow_thinking=thinking,
            max_thinking_tokens=512,
        )
        self.chat = talu.Chat(
            model,
            system=system or "You are a helpful assistant.",
            config=config,
        )

        # Track usage
        self.total_tokens = 0
        self.message_count = 0

        # Interrupt flag for graceful Ctrl+C
        self.interrupted = False

        # Command history (persisted to file)
        self.session = PromptSession(
            history=FileHistory(".talu_chat_history"),
            style=Style.from_dict({
                "prompt": "bold cyan",
            }),
        )

    def handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully during generation."""
        self.interrupted = True
        self.console.print("\n[yellow]Generation interrupted.[/yellow]")

    def print_welcome(self):
        """Print welcome message with commands."""
        self.console.print(Panel(
            "[bold]Interactive Chat[/bold]\n\n"
            f"Model: [cyan]{self.model}[/cyan]\n"
            f"Thinking mode: [cyan]{'enabled' if self.thinking else 'disabled'}[/cyan]\n\n"
            "[dim]Commands:[/dim]\n"
            "  /clear   - Clear conversation history\n"
            "  /history - Show conversation history\n"
            "  /system  - Change system prompt\n"
            "  /think   - Toggle thinking mode\n"
            "  /usage   - Show token usage\n"
            "  /exit    - Exit the chat\n\n"
            "[dim]Press Ctrl+C to interrupt generation, Ctrl+D to exit.[/dim]",
            title="talu",
            border_style="blue",
        ))

    def handle_command(self, cmd: str) -> bool:
        """Handle special commands. Returns True if should continue loop."""
        cmd = cmd.strip().lower()

        if cmd == "/exit":
            return False

        elif cmd == "/clear":
            self.chat.clear()
            self.console.print("[green]Conversation cleared.[/green]")

        elif cmd == "/history":
            if len(self.chat.messages) == 0:
                self.console.print("[dim]No messages yet.[/dim]")
            else:
                for msg in self.chat.messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    color = {"system": "yellow", "user": "cyan", "assistant": "green"}.get(role, "white")
                    self.console.print(f"[bold {color}]{role}:[/bold {color}] {content[:100]}...")

        elif cmd == "/system":
            self.console.print("[dim]Enter new system prompt (or empty to cancel):[/dim]")
            try:
                new_system = self.session.prompt("System: ")
                if new_system.strip():
                    self.chat.reset()
                    self.chat = talu.Chat(
                        self.model,
                        system=new_system,
                        config=self.chat.config,
                    )
                    self.console.print(f"[green]System prompt updated.[/green]")
            except (EOFError, KeyboardInterrupt):
                pass

        elif cmd == "/think":
            self.thinking = not self.thinking
            self.chat.config = GenerationConfig(
                max_tokens=self.max_tokens,
                allow_thinking=self.thinking,
                max_thinking_tokens=512,
            )
            status = "enabled" if self.thinking else "disabled"
            self.console.print(f"[green]Thinking mode {status}.[/green]")

        elif cmd == "/usage":
            self.console.print(
                f"[dim]Messages: {self.message_count} | "
                f"Total tokens: {self.total_tokens}[/dim]"
            )

        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")

        return True

    def stream_response(self, user_input: str):
        """Stream a response with rich formatting."""
        self.interrupted = False

        # Set up interrupt handler
        old_handler = signal.signal(signal.SIGINT, self.handle_interrupt)

        try:
            # Collect the streamed response
            collected = []
            thinking_content = []
            in_thinking = False

            self.console.print("[bold green]Assistant:[/bold green]")

            for chunk in self.chat(user_input):
                if self.interrupted:
                    break

                # Track thinking blocks
                if "<think>" in chunk:
                    in_thinking = True
                    chunk = chunk.replace("<think>", "")
                if "</think>" in chunk:
                    in_thinking = False
                    chunk = chunk.replace("</think>", "")
                    # Print thinking summary
                    if thinking_content:
                        thinking_text = "".join(thinking_content)
                        self.console.print(
                            f"[dim italic]Thinking: {thinking_text[:200]}...[/dim italic]"
                            if len(thinking_text) > 200 else
                            f"[dim italic]Thinking: {thinking_text}[/dim italic]"
                        )
                        thinking_content = []
                    continue

                if in_thinking:
                    thinking_content.append(chunk)
                else:
                    collected.append(chunk)
                    # Print chunk in real-time
                    self.console.print(chunk, end="")

            self.console.print()  # Newline after response

            # Update usage stats
            self.message_count += 1
            if hasattr(self.chat, "_last_response") and self.chat._last_response:  # WORKAROUND: until public usage API
                usage = self.chat._last_response.usage  # WORKAROUND: until public usage API
                if usage:
                    self.total_tokens += usage.total_tokens

        finally:
            # Restore original handler
            signal.signal(signal.SIGINT, old_handler)

    def run(self):
        """Main chat loop."""
        self.print_welcome()

        while True:
            try:
                # Get user input with prompt_toolkit (supports history)
                user_input = self.session.prompt(
                    [("class:prompt", "You: ")],
                ).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                # Stream response
                self.stream_response(user_input)

            except EOFError:
                # Ctrl+D pressed
                break
            except KeyboardInterrupt:
                # Ctrl+C at prompt - just continue
                self.console.print()
                continue

        self.console.print("\n[dim]Goodbye![/dim]")


def main():
    parser = argparse.ArgumentParser(description="Interactive terminal chat")
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen3-0.6B",
        help="Model to use (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--system", "-s",
        default=None,
        help="System prompt",
    )
    parser.add_argument(
        "--thinking", "-t",
        action="store_true",
        help="Enable thinking mode (chain-of-thought)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    args = parser.parse_args()

    chat = InteractiveChat(
        model=args.model,
        system=args.system,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
    )
    chat.run()


if __name__ == "__main__":
    main()

"""
Topics covered:
* chat.streaming
* stream.tokens
* workflow.end.to.end
"""
