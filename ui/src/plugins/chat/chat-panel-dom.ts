/**
 * Chat panel DOM — builds the data-panel content (model, sampling, info, events)
 * programmatically. Singleton: created once, refs persist across show/hide.
 *
 * The root element is passed to layout.showPanel({ content }).
 * Element refs survive DOM detachment (values persist on hidden inputs).
 */

export interface ChatPanelDom {
  root: HTMLElement;
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
  panelEventsVerbosity: HTMLSelectElement;
  panelEventsClear: HTMLButtonElement;
  panelEventsLog: HTMLElement;
}

let cached: ChatPanelDom | null = null;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs?: Record<string, string>,
  children?: (HTMLElement | string)[],
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "className") e.className = v;
      else if (k === "textContent") e.textContent = v;
      else e.setAttribute(k, v);
    }
  }
  if (children) {
    for (const c of children) {
      if (typeof c === "string") e.appendChild(document.createTextNode(c));
      else e.appendChild(c);
    }
  }
  return e;
}

function numberInput(
  id: string,
  label: string,
  opts: { step: string; min: string; max?: string; placeholder: string },
): { wrapper: HTMLElement; input: HTMLInputElement; hint?: HTMLElement } {
  const wrapper = el("div", { style: "margin-bottom: 0.75rem" });
  const lbl = el("label", { className: "form-label form-label-sm", for: id });
  lbl.textContent = label;
  wrapper.appendChild(lbl);

  const input = el("input", {
    id,
    type: "number",
    className: "form-input form-input-sm",
    step: opts.step,
    min: opts.min,
    placeholder: opts.placeholder,
  });
  if (opts.max) input.setAttribute("max", opts.max);
  wrapper.appendChild(input);

  return { wrapper, input };
}

function numberInputWithHint(
  id: string,
  label: string,
  opts: { step: string; min: string; max?: string; placeholder: string },
): { wrapper: HTMLElement; input: HTMLInputElement; hint: HTMLElement } {
  const { wrapper, input } = numberInput(id, label, opts);
  const hint = el("div", { id: `${id}-default`, className: "form-hint" });
  wrapper.appendChild(hint);
  return { wrapper, input, hint };
}

export function getChatPanelDom(): ChatPanelDom {
  if (cached) return cached;

  const root = el("div", { className: "panel-section" });

  // ── Model select ──────────────────────────────────────────────────────
  const modelWrapper = el("div", { style: "margin-bottom: 0.75rem" });
  const modelLabel = el("label", { className: "form-label form-label-sm", for: "panel-model" });
  modelLabel.textContent = "Model";
  modelWrapper.appendChild(modelLabel);

  const panelModel = el("select", { id: "panel-model", className: "form-select form-select-sm" });
  const loadingOpt = el("option", { value: "" });
  loadingOpt.textContent = "Loading...";
  panelModel.appendChild(loadingOpt);
  modelWrapper.appendChild(panelModel);
  root.appendChild(modelWrapper);

  // ── Sampling section ──────────────────────────────────────────────────
  root.appendChild(el("div", { className: "panel-divider" }));
  const samplingHeading = el("h3", { className: "panel-heading" });
  samplingHeading.textContent = "Sampling";
  root.appendChild(samplingHeading);

  const temp = numberInputWithHint("panel-temperature", "Temperature", { step: "0.1", min: "0", max: "2", placeholder: "1.0" });
  root.appendChild(temp.wrapper);

  const topP = numberInputWithHint("panel-top-p", "Top P", { step: "0.05", min: "0", max: "1", placeholder: "1.0" });
  root.appendChild(topP.wrapper);

  const topK = numberInputWithHint("panel-top-k", "Top K", { step: "1", min: "0", placeholder: "50" });
  root.appendChild(topK.wrapper);

  const minP = numberInput("panel-min-p", "Min P", { step: "0.01", min: "0", max: "1", placeholder: "0.0" });
  root.appendChild(minP.wrapper);

  const maxOut = numberInput("panel-max-output-tokens", "Max Output Tokens", { step: "1", min: "1", placeholder: "2048" });
  root.appendChild(maxOut.wrapper);

  const repPen = numberInput("panel-repetition-penalty", "Repetition Penalty", { step: "0.05", min: "1", placeholder: "1.0" });
  root.appendChild(repPen.wrapper);

  const seed = numberInput("panel-seed", "Seed", { step: "1", min: "0", placeholder: "Random" });
  root.appendChild(seed.wrapper);

  // ── Info section ──────────────────────────────────────────────────────
  root.appendChild(el("div", { className: "panel-divider" }));
  const infoHeading = el("h3", { className: "panel-heading" });
  infoHeading.textContent = "Info";
  root.appendChild(infoHeading);

  const panelChatInfo = el("div");

  const createdRow = el("div", { className: "info-row" });
  const createdLabel = el("span", { className: "info-label" });
  createdLabel.textContent = "Created";
  const panelInfoCreated = el("span", { className: "info-value" });
  panelInfoCreated.textContent = "-";
  createdRow.appendChild(createdLabel);
  createdRow.appendChild(panelInfoCreated);
  panelChatInfo.appendChild(createdRow);

  const panelInfoForkedRow = el("div", { className: "info-row hidden" });
  const forkedLabel = el("span", { className: "info-label" });
  forkedLabel.textContent = "Forked from";
  const panelInfoForked = el("span", { className: "info-value mono" });
  panelInfoForked.textContent = "-";
  panelInfoForkedRow.appendChild(forkedLabel);
  panelInfoForkedRow.appendChild(panelInfoForked);
  panelChatInfo.appendChild(panelInfoForkedRow);

  root.appendChild(panelChatInfo);

  // ── Events section ────────────────────────────────────────────────────
  root.appendChild(el("div", { className: "panel-divider" }));
  const eventsHeading = el("h3", { className: "panel-heading" });
  eventsHeading.textContent = "Events";
  root.appendChild(eventsHeading);

  const eventsControls = el("div", { className: "chat-events-controls" });

  const panelEventsVerbosity = el("select", { className: "form-select form-select-sm", title: "Events verbosity" });
  for (const [val, text] of [["1", "v"], ["2", "vv"], ["3", "vvv"]] as const) {
    const opt = el("option", { value: val });
    opt.textContent = text;
    panelEventsVerbosity.appendChild(opt);
  }
  eventsControls.appendChild(panelEventsVerbosity);

  const panelEventsClear = el("button", { className: "btn btn-ghost btn-icon", title: "Clear events" }) as HTMLButtonElement;
  panelEventsClear.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>';
  eventsControls.appendChild(panelEventsClear);
  root.appendChild(eventsControls);

  const panelEventsLog = el("div", { className: "chat-events-log" });
  panelEventsLog.setAttribute("aria-live", "polite");
  root.appendChild(panelEventsLog);

  cached = {
    root,
    panelModel,
    panelTemperature: temp.input,
    panelTopP: topP.input,
    panelTopK: topK.input,
    panelMinP: minP.input,
    panelMaxOutputTokens: maxOut.input,
    panelRepetitionPenalty: repPen.input,
    panelSeed: seed.input,
    panelTemperatureDefault: temp.hint,
    panelTopPDefault: topP.hint,
    panelTopKDefault: topK.hint,
    panelChatInfo,
    panelInfoCreated,
    panelInfoForkedRow,
    panelInfoForked,
    panelEventsVerbosity,
    panelEventsClear,
    panelEventsLog,
  };

  return cached;
}
