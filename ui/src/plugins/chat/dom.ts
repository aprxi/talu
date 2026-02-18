/**
 * Chat plugin DOM cache — queries elements within the plugin's
 * shadow root container (set via initChatDom).
 */

export interface ChatDom {
  // Sidebar
  sidebarList: HTMLElement;
  sidebarSentinel: HTMLElement;
  newConversationBtn: HTMLButtonElement;

  // Transcript
  transcriptContainer: HTMLElement;

  // Welcome state
  welcomeState: HTMLElement;
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

  // Right panel
  rightPanel: HTMLElement;
  closeRightPanelBtn: HTMLButtonElement;
  panelModel: HTMLSelectElement;
  panelTemperature: HTMLInputElement;
  panelTopP: HTMLInputElement;
  panelTopK: HTMLInputElement;
  panelMinP: HTMLInputElement;
  panelMaxOutputTokens: HTMLInputElement;
  panelRepetitionPenalty: HTMLInputElement;
  panelSeed: HTMLInputElement;
  panelTemperatureDefault: HTMLElement;
  panelTopPDefault: HTMLElement;
  panelTopKDefault: HTMLElement;
  panelChatInfo: HTMLElement;
  panelInfoCreated: HTMLElement;
  panelInfoForkedRow: HTMLElement;
  panelInfoForked: HTMLElement;
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
    newConversationBtn: q<HTMLButtonElement>("#new-conversation"),

    transcriptContainer: q("#transcript"),

    welcomeState: q("#welcome-state"),
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

    rightPanel: q("#right-panel"),
    closeRightPanelBtn: q<HTMLButtonElement>("#close-right-panel"),
    panelModel: q<HTMLSelectElement>("#panel-model"),
    panelTemperature: q<HTMLInputElement>("#panel-temperature"),
    panelTopP: q<HTMLInputElement>("#panel-top-p"),
    panelTopK: q<HTMLInputElement>("#panel-top-k"),
    panelMinP: q<HTMLInputElement>("#panel-min-p"),
    panelMaxOutputTokens: q<HTMLInputElement>("#panel-max-output-tokens"),
    panelRepetitionPenalty: q<HTMLInputElement>("#panel-repetition-penalty"),
    panelSeed: q<HTMLInputElement>("#panel-seed"),
    panelTemperatureDefault: q("#panel-temperature-default"),
    panelTopPDefault: q("#panel-top-p-default"),
    panelTopKDefault: q("#panel-top-k-default"),
    panelChatInfo: q("#panel-chat-info"),
    panelInfoCreated: q("#panel-info-created"),
    panelInfoForkedRow: q("#panel-info-forked-row"),
    panelInfoForked: q("#panel-info-forked"),
  };

  return cached;
}
