interface TaluBootstrap {
  workdir?: string | null;
}

function readBootstrap(): TaluBootstrap | null {
  const value = (globalThis as { __TALU_BOOTSTRAP__?: unknown }).__TALU_BOOTSTRAP__;
  if (!value || typeof value !== "object") return null;
  return value as TaluBootstrap;
}

export function getBootstrapWorkdir(): string | null {
  const workdir = readBootstrap()?.workdir;
  if (typeof workdir !== "string" || workdir.length === 0) return null;
  return workdir;
}
