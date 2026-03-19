import { bootKernel } from "./kernel/core/plugin-manager.ts";
import { settingsPlugin } from "./plugins/settings/index.ts";
import { promptsPlugin } from "./plugins/prompts/index.ts";
import { browserPlugin } from "./plugins/browser/index.ts";
import { filesPlugin } from "./plugins/files/index.ts";
import { chatPlugin } from "./plugins/chat/index.ts";
import { repoPlugin } from "./plugins/repo/index.ts";
import { editorOpsPlugin } from "./plugins/editor-ops/index.ts";
import { bootBenchApp } from "./bench/app.ts";

document.addEventListener("DOMContentLoaded", () => {
  if (
    window.location.pathname === "/bench" ||
    window.location.pathname.startsWith("/bench/")
  ) {
    bootBenchApp();
    return;
  }

  bootKernel([
    settingsPlugin,
    promptsPlugin,
    browserPlugin,
    filesPlugin,
    editorOpsPlugin,
    chatPlugin,
    repoPlugin,
  ]);
});
