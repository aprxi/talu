import { bootKernel } from "./kernel/core/plugin-manager.ts";
import { settingsPlugin } from "./plugins/settings/index.ts";
import { promptsPlugin } from "./plugins/prompts/index.ts";
import { browserPlugin } from "./plugins/browser/index.ts";
import { filesPlugin } from "./plugins/files/index.ts";
import { chatPlugin } from "./plugins/chat/index.ts";
import { repoPlugin } from "./plugins/repo/index.ts";

document.addEventListener("DOMContentLoaded", () => {
  bootKernel([settingsPlugin, promptsPlugin, browserPlugin, filesPlugin, chatPlugin, repoPlugin]);
});
