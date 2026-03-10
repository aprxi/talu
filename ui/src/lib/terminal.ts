/** Shared terminal creation utilities. */

import type { Disposable } from "../kernel/types.ts";

export interface TerminalHandle {
  write(text: string): void;
  writeln(text: string): void;
  focus(): void;
  fit(): void;
  getCols(): number;
  getRows(): number;
  onData(handler: (data: string) => void): Disposable;
  dispose(): void;
}

export function createFallbackTerminal(host: HTMLElement): TerminalHandle {
  host.innerHTML = "";
  host.tabIndex = 0;

  const pre = document.createElement("pre");
  pre.style.margin = "0";
  pre.style.padding = "12px";
  pre.style.whiteSpace = "pre-wrap";
  pre.style.wordBreak = "break-word";
  pre.style.height = "100%";
  pre.style.overflow = "auto";
  host.appendChild(pre);

  const listeners = new Set<(data: string) => void>();
  const emitData = (data: string): void => {
    for (const listener of listeners) {
      listener(data);
    }
  };

  const keyHandler = (event: KeyboardEvent): void => {
    if (event.key === "Tab") {
      event.preventDefault();
      emitData("\t");
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      emitData("\r");
      return;
    }
    if (event.key === "Backspace") {
      emitData("\u007f");
      return;
    }
    if (event.ctrlKey && (event.key === "c" || event.key === "C")) {
      event.preventDefault();
      emitData("\u0003");
      return;
    }
    if (event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) {
      emitData(event.key);
    }
  };

  host.addEventListener("keydown", keyHandler);

  return {
    write(text: string): void {
      pre.textContent = `${pre.textContent ?? ""}${text}`;
      pre.scrollTop = pre.scrollHeight;
    },
    writeln(text: string): void {
      pre.textContent = `${pre.textContent ?? ""}${text}\n`;
      pre.scrollTop = pre.scrollHeight;
    },
    focus(): void {
      host.focus();
    },
    fit(): void {
      // no-op
    },
    getCols(): number {
      return 120;
    },
    getRows(): number {
      return 32;
    },
    onData(handler: (data: string) => void): Disposable {
      listeners.add(handler);
      return {
        dispose(): void {
          listeners.delete(handler);
        },
      };
    },
    dispose(): void {
      host.removeEventListener("keydown", keyHandler);
      listeners.clear();
      host.innerHTML = "";
    },
  };
}

export async function createTerminal(host: HTMLElement): Promise<TerminalHandle> {
  try {
    const [{ Terminal }, { FitAddon }] = await Promise.all([
      import("xterm"),
      import("xterm-addon-fit"),
    ]);

    host.innerHTML = "";
    const terminal = new Terminal({
      cursorBlink: true,
      scrollback: 10_000,
      fontSize: 13,
      fontFamily:
        "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace",
    });
    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(host);
    fitAddon.fit();

    return {
      write(text: string): void {
        terminal.write(text);
      },
      writeln(text: string): void {
        terminal.writeln(text);
      },
      focus(): void {
        terminal.focus();
      },
      fit(): void {
        fitAddon.fit();
      },
      getCols(): number {
        return terminal.cols;
      },
      getRows(): number {
        return terminal.rows;
      },
      onData(handler: (data: string) => void): Disposable {
        const sub = terminal.onData(handler);
        return {
          dispose(): void {
            sub.dispose();
          },
        };
      },
      dispose(): void {
        terminal.dispose();
      },
    };
  } catch (error) {
    console.warn("[terminal] xterm unavailable, using fallback terminal", error);
    return createFallbackTerminal(host);
  }
}
