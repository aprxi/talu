import type {
  AgentAccess,
  AgentExecEvent,
  AgentExecOptions,
  AgentExecResult,
  AgentProcessEvent,
  AgentProcessOpenOptions,
  AgentProcessSession,
  AgentShellEvent,
  AgentShellOpenOptions,
  AgentShellSession,
  Disposable,
} from "../types.ts";
import type { ApiClient, ApiResult } from "../../api.ts";

interface CreateAgentAccessOptions {
  api: ApiClient;
  requirePermission: (name: "filesystem" | "exec") => void;
  defaultCwd?: string | null;
}

function unwrapApiResult<T>(operation: string, result: ApiResult<T>): T {
  if (result.ok && result.data !== undefined) {
    return result.data;
  }
  throw new Error(result.error ? `Agent ${operation} failed: ${result.error}` : `Agent ${operation} failed`);
}

async function parseErrorMessage(resp: Response): Promise<string> {
  const err = await resp.json().catch(() => null);
  const code = err?.error?.code;
  const message = err?.error?.message ?? `${resp.status} ${resp.statusText}`;
  return code ? `${code}: ${message}` : message;
}

function parseExecEvent(raw: unknown): AgentExecEvent | null {
  if (!raw || typeof raw !== "object") return null;

  const value = raw as { type?: unknown; data?: unknown; code?: unknown; message?: unknown };
  const type = typeof value.type === "string" ? value.type : null;
  if (type !== "stdout" && type !== "stderr" && type !== "exit" && type !== "error") {
    return null;
  }

  return {
    type,
    data: typeof value.data === "string" ? value.data : undefined,
    code: typeof value.code === "number" ? value.code : undefined,
    message: typeof value.message === "string" ? value.message : undefined,
  };
}

async function readExecStream(
  resp: Response,
  opts: AgentExecOptions | undefined,
): Promise<AgentExecResult> {
  if (!resp.body) {
    throw new Error("Agent shell exec failed: response body is empty");
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let stdout = "";
  let stderr = "";
  let exitCode: number | null = null;
  const events: AgentExecEvent[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data) continue;

      let payload: unknown;
      try {
        payload = JSON.parse(data);
      } catch {
        continue;
      }

      const event = parseExecEvent(payload);
      if (!event) continue;

      events.push(event);
      opts?.onEvent?.(event);

      if (event.type === "stdout") {
        stdout += event.data ?? "";
      } else if (event.type === "stderr") {
        stderr += event.data ?? "";
      } else if (event.type === "exit") {
        exitCode = event.code ?? null;
      } else if (event.type === "error") {
        throw new Error(event.message ? `Agent shell exec failed: ${event.message}` : "Agent shell exec failed");
      }
    }
  }

  return {
    stdout,
    stderr,
    exitCode,
    events,
  };
}

function toWebSocketUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

export function createAgentAccess(options: CreateAgentAccessOptions): AgentAccess {
  const { api, requirePermission } = options;
  const encoder = new TextEncoder();

  const agent: AgentAccess = {
    cwd: options.defaultCwd ?? null,
    fs: {
      async readFile(path, opts) {
        requirePermission("filesystem");
        const data = unwrapApiResult(
          "fs.readFile",
          await api.agentFsRead({
            path,
            encoding: opts?.encoding,
            max_bytes: opts?.maxBytes,
          }),
        );
        return {
          path: data.path,
          content: data.content,
          encoding: data.encoding,
          size: data.size,
          truncated: data.truncated,
        };
      },

      async writeFile(path, content, opts) {
        requirePermission("filesystem");
        const data = unwrapApiResult(
          "fs.writeFile",
          await api.agentFsWrite({
            path,
            content,
            encoding: opts?.encoding,
            mkdir: opts?.mkdir,
          }),
        );
        return {
          path: data.path,
          bytesWritten: data.bytes_written,
        };
      },

      async editFile(path, oldText, newText, opts) {
        requirePermission("filesystem");
        const data = unwrapApiResult(
          "fs.editFile",
          await api.agentFsEdit({
            path,
            old_text: oldText,
            new_text: newText,
            replace_all: opts?.replaceAll,
          }),
        );
        return {
          path: data.path,
          replacements: data.replacements,
        };
      },

      async stat(path) {
        requirePermission("filesystem");
        const data = unwrapApiResult("fs.stat", await api.agentFsStat({ path }));
        return {
          path: data.path,
          exists: data.exists,
          isFile: data.is_file,
          isDir: data.is_dir,
          isSymlink: data.is_symlink,
          size: data.size,
          mode: data.mode,
          modifiedAt: data.modified_at,
          createdAt: data.created_at,
        };
      },

      async ls(path, opts) {
        requirePermission("filesystem");
        const data = unwrapApiResult(
          "fs.ls",
          await api.agentFsList({
            path,
            glob: opts?.glob,
            recursive: opts?.recursive,
            limit: opts?.limit,
          }),
        );
        return {
          path: data.path,
          truncated: data.truncated,
          entries: data.entries.map((entry) => ({
            name: entry.name,
            path: entry.path,
            isDir: entry.is_dir,
            isSymlink: entry.is_symlink,
            size: entry.size,
            modifiedAt: entry.modified_at,
          })),
        };
      },

      async rm(path, opts) {
        requirePermission("filesystem");
        unwrapApiResult(
          "fs.rm",
          await api.agentFsRemove({
            path,
            recursive: opts?.recursive,
          }),
        );
      },

      async mkdir(path) {
        requirePermission("filesystem");
        unwrapApiResult("fs.mkdir", await api.agentFsMkdir({ path, recursive: true }));
      },

      async rename(from, to) {
        requirePermission("filesystem");
        unwrapApiResult("fs.rename", await api.agentFsRename({ from, to }));
      },
    },

    shell: {
      async exec(command, opts) {
        requirePermission("exec");
        if (command.trim().length === 0) {
          throw new Error("Agent shell exec failed: command must be non-empty");
        }

        const resp = await api.agentExec(
          {
            command,
            cwd: opts?.cwd ?? agent.cwd ?? undefined,
            timeout_ms: opts?.timeoutMs,
          },
          opts?.signal,
        );

        if (!resp.ok) {
          const message = await parseErrorMessage(resp);
          throw new Error(`Agent shell exec failed: ${message}`);
        }

        return readExecStream(resp, opts);
      },

      async open(opts?: AgentShellOpenOptions): Promise<AgentShellSession> {
        requirePermission("exec");

        const created = unwrapApiResult(
          "shell.open",
          await api.agentShellCreate({
            cwd: opts?.cwd ?? agent.cwd ?? undefined,
            cols: opts?.cols,
            rows: opts?.rows,
          }),
        );

        const listeners = new Set<(event: AgentShellEvent) => void>();
        let closed = false;
        const decoder = new TextDecoder();

        const emit = (event: AgentShellEvent): void => {
          opts?.onEvent?.(event);
          for (const handler of listeners) {
            handler(event);
          }
        };

        const ws = new WebSocket(toWebSocketUrl(`/v1/agent/shells/${encodeURIComponent(created.shell_id)}/ws`));
        ws.binaryType = "arraybuffer";

        ws.addEventListener("message", (evt) => {
          if (typeof evt.data === "string") {
            let parsed: { type?: string; code?: number; message?: string } | null = null;
            try {
              parsed = JSON.parse(evt.data) as { type?: string; code?: number; message?: string };
            } catch {
              parsed = null;
            }
            if (!parsed) return;
            if (parsed.type === "exit") {
              emit({ type: "exit", code: parsed.code });
            } else if (parsed.type === "error") {
              emit({ type: "error", message: parsed.message ?? "shell error" });
            }
            return;
          }

          if (evt.data instanceof ArrayBuffer) {
            emit({ type: "data", data: decoder.decode(evt.data) });
            return;
          }

          if (evt.data instanceof Blob) {
            void evt.data.arrayBuffer().then((buffer) => {
              emit({ type: "data", data: decoder.decode(buffer) });
            });
          }
        });

        ws.addEventListener("error", () => {
          emit({ type: "error", message: "shell websocket error" });
        });

        ws.addEventListener("close", () => {
          emit({ type: "exit" });
        });

        return {
          id: created.shell_id,
          cols: created.cols,
          rows: created.rows,
          cwd: created.cwd ?? null,
          send(data: string): void {
            if (ws.readyState !== WebSocket.OPEN) {
              throw new Error("Agent shell send failed: websocket is not open");
            }
            ws.send(encoder.encode(data));
          },
          resize(cols: number, rows: number): void {
            if (ws.readyState !== WebSocket.OPEN) {
              throw new Error("Agent shell resize failed: websocket is not open");
            }
            ws.send(JSON.stringify({ type: "resize", cols, rows }));
          },
          signal(name: string): void {
            if (ws.readyState !== WebSocket.OPEN) {
              throw new Error("Agent shell signal failed: websocket is not open");
            }
            ws.send(JSON.stringify({ type: "signal", signal: name }));
          },
          onEvent(handler: (event: AgentShellEvent) => void): Disposable {
            listeners.add(handler);
            return {
              dispose(): void {
                listeners.delete(handler);
              },
            };
          },
          async close(): Promise<void> {
            if (closed) return;
            closed = true;

            if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
              ws.close();
            }

            const result = await api.agentShellDelete(created.shell_id);
            if (!result.ok && !result.error?.includes("shell not found")) {
              throw new Error(`Agent shell close failed: ${result.error ?? "unknown error"}`);
            }
          },
        };
      },
    },

    process: {
      async open(command: string, opts?: AgentProcessOpenOptions): Promise<AgentProcessSession> {
        requirePermission("exec");
        if (!command.trim()) {
          throw new Error("Agent process open failed: command is empty");
        }

        const created = unwrapApiResult(
          "process.open",
          await api.agentProcessSpawn({
            command,
            cwd: opts?.cwd ?? agent.cwd ?? undefined,
          }),
        );

        const listeners = new Set<(event: AgentProcessEvent) => void>();
        let closed = false;
        let exited = false;
        const decoder = new TextDecoder();

        const emit = (event: AgentProcessEvent): void => {
          opts?.onEvent?.(event);
          for (const handler of listeners) {
            handler(event);
          }
        };

        const ws = new WebSocket(toWebSocketUrl(`/v1/agent/processes/${encodeURIComponent(created.process_id)}/ws`));
        ws.binaryType = "arraybuffer";

        ws.addEventListener("message", (evt) => {
          if (typeof evt.data === "string") {
            let parsed: { type?: string; code?: number; message?: string } | null = null;
            try {
              parsed = JSON.parse(evt.data) as { type?: string; code?: number; message?: string };
            } catch {
              parsed = null;
            }
            if (!parsed) return;
            if (parsed.type === "exit") {
              exited = true;
              emit({ type: "exit", code: parsed.code });
            } else if (parsed.type === "error") {
              emit({ type: "error", message: parsed.message ?? "process error" });
            }
            return;
          }

          if (evt.data instanceof ArrayBuffer) {
            emit({ type: "data", data: decoder.decode(evt.data) });
            return;
          }

          if (evt.data instanceof Blob) {
            void evt.data.arrayBuffer().then((buffer) => {
              emit({ type: "data", data: decoder.decode(buffer) });
            });
          }
        });

        ws.addEventListener("error", () => {
          emit({ type: "error", message: "process websocket error" });
        });

        ws.addEventListener("close", () => {
          if (!exited) emit({ type: "exit" });
        });

        return {
          id: created.process_id,
          command,
          cwd: created.cwd ?? null,
          send(data: string): void {
            if (ws.readyState !== WebSocket.OPEN) {
              throw new Error("Agent process send failed: websocket is not open");
            }
            ws.send(encoder.encode(data));
          },
          signal(name: string): void {
            if (ws.readyState !== WebSocket.OPEN) {
              throw new Error("Agent process signal failed: websocket is not open");
            }
            ws.send(JSON.stringify({ type: "signal", signal: name }));
          },
          onEvent(handler: (event: AgentProcessEvent) => void): Disposable {
            listeners.add(handler);
            return {
              dispose(): void {
                listeners.delete(handler);
              },
            };
          },
          async close(): Promise<void> {
            if (closed) return;
            closed = true;

            if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
              ws.close();
            }

            const result = await api.agentProcessDelete(created.process_id);
            if (!result.ok && !result.error?.includes("process not found")) {
              throw new Error(`Agent process close failed: ${result.error ?? "unknown error"}`);
            }
          },
        };
      },
    },
  };

  return agent;
}
