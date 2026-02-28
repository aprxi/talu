/**
 * Chat plugin DOM cache — queries elements within the plugin's
 * shadow root container (set via initChatDom).
 */

export interface ChatDom {
  // Sidebar
  sidebarList: HTMLElement;
  sidebarSentinel: HTMLElement;
  sidebarSearch: HTMLInputElement;
  sidebarSearchClear: HTMLButtonElement;
  sidebarNewProject: HTMLButtonElement;
  sidebarCollapseAll: HTMLButtonElement;
  sidebarSort: HTMLButtonElement;

  // Transcript
  transcriptContainer: HTMLElement;

  // Welcome state
  welcomeState: HTMLElement;
  welcomeProject: HTMLElement;
  welcomeInput: HTMLTextAreaElement;
  welcomeSend: HTMLButtonElement;
  welcomeAttach: HTMLButtonElement;
  welcomeLibrary: HTMLButtonElement;
  welcomeAttachmentList: HTMLElement;
  welcomeModel: HTMLSelectElement;
  welcomePrompt: HTMLSelectElement;

  // Active chat input
  inputBar: HTMLElement;
  inputText: HTMLTextAreaElement;
  inputSend: HTMLButtonElement;
  inputAttach: HTMLButtonElement;
  inputLibrary: HTMLButtonElement;
  inputAttachmentList: HTMLElement;
  fileInput: HTMLInputElement;

}

let root: HTMLElement;
let cached: ChatDom | null = null;

/** Set the root container for DOM queries. Must be called before getChatDom(). */
export function initChatDom(container: HTMLElement): void {
  root = container;
  cached = null;
}

export function getChatDom(): ChatDom {
  if (cached) return cached;

  const q = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;

  cached = {
    sidebarList: q("#sidebar-list"),
    sidebarSentinel: q("#loader-sentinel"),
    sidebarSearch: q<HTMLInputElement>("#sidebar-search"),
    sidebarSearchClear: q<HTMLButtonElement>("#sidebar-search-clear"),
    sidebarNewProject: q<HTMLButtonElement>("#sidebar-new-project-btn"),
    sidebarCollapseAll: q<HTMLButtonElement>("#sidebar-collapse-all-btn"),
    sidebarSort: q<HTMLButtonElement>("#sidebar-sort-btn"),

    transcriptContainer: q("#transcript"),

    welcomeState: q("#welcome-state"),
    welcomeProject: q("#welcome-project"),
    welcomeInput: q<HTMLTextAreaElement>("#welcome-input"),
    welcomeSend: q<HTMLButtonElement>("#welcome-send"),
    welcomeAttach: q<HTMLButtonElement>("#welcome-attach"),
    welcomeLibrary: q<HTMLButtonElement>("#welcome-library"),
    welcomeAttachmentList: q("#welcome-attachment-list"),
    welcomeModel: q<HTMLSelectElement>("#welcome-model"),
    welcomePrompt: q<HTMLSelectElement>("#welcome-prompt"),

    inputBar: q("#input-bar"),
    inputText: q<HTMLTextAreaElement>("#input-text"),
    inputSend: q<HTMLButtonElement>("#input-send"),
    inputAttach: q<HTMLButtonElement>("#input-attach"),
    inputLibrary: q<HTMLButtonElement>("#input-library"),
    inputAttachmentList: q("#input-attachment-list"),
    fileInput: q<HTMLInputElement>("#chat-file-input"),

  };

  return cached;
}
