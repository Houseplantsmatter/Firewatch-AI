document.addEventListener("DOMContentLoaded", () => {
  "use strict";
  const core = window.FireWatchCore;
  const GENERIC_ANALYSIS_ERROR = "The image could not be analyzed. Please try again.";
  const byId = id => document.getElementById(id);
  const state = {file: null, imageUrl: "", thumbnail: "", result: null, history: [], filter: "all", storageAvailable: true, historyError: "", selectionGeneration: 0};
  const nodes = {
    analyzeView: byId("analyze-view"), historyView: byId("history-view"), uploadPanel: byId("upload-panel"),
    fileInput: byId("file-input"), dropZone: byId("drop-zone"), uploadError: byId("upload-error"),
    previewPanel: byId("preview-panel"), previewImage: byId("preview-image"), fileName: byId("file-name"),
    fileSize: byId("file-size"), replaceButton: byId("replace-button"), analyzeButton: byId("analyze-button"),
    progressPanel: byId("progress-panel"), resultPanel: byId("result-panel"), resultImage: byId("result-image"),
    detectionBox: byId("detection-box"), boxLabel: byId("box-label"), resultKicker: byId("result-kicker"),
    resultTitle: byId("result-title"), resultTitleText: byId("result-title-text"), resultIcon: byId("result-icon"), resultDescription: byId("result-description"),
    resultConfidence: byId("result-confidence"), resultCount: byId("result-count"), resultTime: byId("result-time"),
    resultGuidance: byId("result-guidance"), resultStorageMessage: byId("result-storage-message"), analyzeAnother: byId("analyze-another-button"),
    historyCount: byId("history-count"), historyList: byId("history-list"), historyMessage: byId("history-message"),
    clearHistory: byId("clear-history-button"), liveRegion: byId("live-region"), filters: byId("history-filters"),
  };

  function announce(message) { nodes.liveRegion.textContent = ""; window.setTimeout(() => { nodes.liveRegion.textContent = message; }, 20); }
  function setUploadError(message) { nodes.uploadError.textContent = message; nodes.uploadError.hidden = !message; }
  function clearResultStorageMessage() { nodes.resultStorageMessage.textContent = ""; nodes.resultStorageMessage.hidden = true; }
  function formatBytes(bytes) { return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`; }
  function setView(name) {
    const history = name === "history";
    nodes.analyzeView.hidden = history; nodes.historyView.hidden = !history;
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === name && button.classList.contains("nav-button");
      button.classList.toggle("is-active", active);
      if (button.classList.contains("nav-button")) active ? button.setAttribute("aria-current", "page") : button.removeAttribute("aria-current");
    });
    window.history.replaceState(null, "", history ? "#history" : "#analyze");
    if (history) renderHistory();
  }
  function decodeImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file); const image = new Image();
      image.onload = () => resolve({image, url});
      image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("decode")); };
      image.src = url;
    });
  }
  function makeThumbnail(image) {
    const scale = Math.min(1, 320 / image.naturalWidth, 200 / image.naturalHeight);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale)); canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.72);
  }
  async function selectFile(file) {
    const generation = ++state.selectionGeneration;
    setUploadError("");
    const error = core.validateFile(file); if (error) { setUploadError(error); return; }
    try {
      const decoded = await decodeImage(file);
      if (generation !== state.selectionGeneration) { URL.revokeObjectURL(decoded.url); return; }
      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
      state.file = file; state.imageUrl = decoded.url; state.thumbnail = makeThumbnail(decoded.image);
      nodes.previewImage.src = state.imageUrl; nodes.fileName.textContent = file.name; nodes.fileSize.textContent = formatBytes(file.size);
      nodes.dropZone.hidden = true; nodes.previewPanel.hidden = false; nodes.analyzeButton.disabled = false;
      announce(`${file.name} is ready to analyze.`);
    } catch (_) { if (generation === state.selectionGeneration) setUploadError("This image could not be read. Choose a different file."); }
  }
  function resetAnalyze() {
    state.selectionGeneration += 1;
    state.file = null; state.result = null; state.thumbnail = "";
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl); state.imageUrl = "";
    nodes.fileInput.value = ""; nodes.previewImage.removeAttribute("src"); nodes.resultImage.removeAttribute("src");
    nodes.previewPanel.hidden = true; nodes.dropZone.hidden = false; nodes.progressPanel.hidden = true; nodes.resultPanel.hidden = true; nodes.uploadPanel.hidden = false;
    nodes.analyzeButton.disabled = false; setUploadError(""); nodes.fileInput.focus();
  }
  function copyFor(category) {
    if (category === "fire") return {title: "Possible fire detected.", kicker: "Review needed", description: "A possible flame pattern was identified in this image.", guidance: "Check the original scene. If there is immediate danger, move to safety and call emergency services."};
    if (category === "smoke") return {title: "Possible smoke detected.", kicker: "Review needed", description: "A possible smoke pattern was identified in this image.", guidance: "Check the original scene. If there is immediate danger, move to safety and call emergency services."};
    return {title: "No concern found.", kicker: "Analysis complete", description: "No fire or smoke region was found at or above the model's detection threshold.", guidance: "This result cannot confirm that a real scene is safe. Continue to use working smoke alarms and official emergency information."};
  }
  function categoryIcon(category) { return category === "fire" ? "▲" : category === "smoke" ? "☁" : "✓"; }
  function renderResult(result, imageUrl) {
    const copy = copyFor(result.category); state.result = result; nodes.resultPanel.dataset.category = result.category;
    nodes.resultImage.src = imageUrl; nodes.resultKicker.textContent = copy.kicker; nodes.resultIcon.textContent = categoryIcon(result.category); nodes.resultTitleText.textContent = copy.title;
    nodes.resultDescription.textContent = copy.description; nodes.resultGuidance.textContent = copy.guidance;
    nodes.resultConfidence.textContent = result.confidence === null ? "\u2014" : `${Math.round(result.confidence * 100)}%`;
    nodes.resultCount.textContent = String(result.detectionCount); nodes.resultTime.textContent = new Date(result.timestamp).toLocaleString();
    nodes.resultTitle.tabIndex = -1;
    nodes.detectionBox.hidden = !result.box;
    if (result.box) {
      Object.assign(nodes.detectionBox.style, {left: `${result.box.x * 100}%`, top: `${result.box.y * 100}%`, width: `${result.box.width * 100}%`, height: `${result.box.height * 100}%`});
      nodes.boxLabel.textContent = `${result.category.toUpperCase()} \u00b7 ${Math.round(result.confidence * 100)}%`;
    }
    nodes.uploadPanel.hidden = true; nodes.progressPanel.hidden = true; nodes.resultPanel.hidden = false;
  }
  function loadHistory() {
    let stored;
    try { stored = localStorage.getItem(core.STORAGE_KEY); }
    catch (_) { state.history = []; state.storageAvailable = false; return; }
    if (!stored) { state.history = []; state.storageAvailable = true; return; }
    try { state.history = core.normalizeHistory(JSON.parse(stored)); state.storageAvailable = true; }
    catch (_) {
      state.history = [];
      try { localStorage.removeItem(core.STORAGE_KEY); state.storageAvailable = true; }
      catch (_) { state.storageAvailable = false; }
    }
  }
  function saveHistory(result) {
    const candidateHistory = core.normalizeHistory([result, ...state.history.filter(item => item.id !== result.id)]);
    let storageMessage = "";
    try { localStorage.setItem(core.STORAGE_KEY, JSON.stringify(candidateHistory)); state.history = candidateHistory; state.storageAvailable = true; state.historyError = ""; clearResultStorageMessage(); }
    catch (_) { storageMessage = "History could not be saved on this device. Your current result is still available."; state.storageAvailable = false; state.historyError = storageMessage; nodes.resultStorageMessage.textContent = storageMessage; nodes.resultStorageMessage.hidden = false; }
    renderHistory();
    return storageMessage;
  }
  function historyCard(result) {
    const article = document.createElement("article"); article.className = `history-card ${result.category}`;
    const image = document.createElement("img"); image.src = result.thumbnail; image.alt = "";
    const copy = copyFor(result.category); const details = document.createElement("div"); const title = document.createElement("h2");
    const icon = document.createElement("span"); icon.className = "category-icon"; icon.setAttribute("aria-hidden", "true"); icon.textContent = categoryIcon(result.category);
    const titleText = document.createElement("span"); titleText.textContent = copy.title; title.append(icon, titleText); const meta = document.createElement("p"); meta.textContent = `${new Date(result.timestamp).toLocaleString()} \u00b7 ${result.filename}`;
    details.append(title, meta); const score = document.createElement("strong"); score.textContent = result.confidence === null ? "Clear" : `${Math.round(result.confidence * 100)}%`;
    const actions = document.createElement("div"); actions.className = "history-actions";
    const open = document.createElement("button"); open.type = "button"; open.className = "button secondary"; open.dataset.openId = result.id; open.setAttribute("aria-label", `Open ${result.filename}`); open.textContent = "Open";
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "button text-button"; remove.dataset.deleteId = result.id; remove.setAttribute("aria-label", `Delete ${result.filename}`); remove.textContent = "Delete";
    actions.append(open, remove); article.append(image, details, score, actions); return article;
  }
  function renderHistory() {
    nodes.historyList.replaceChildren(); nodes.historyCount.textContent = String(state.history.length); nodes.clearHistory.disabled = state.history.length === 0;
    document.querySelectorAll("[data-filter]").forEach(button => { const active = button.dataset.filter === state.filter; button.classList.toggle("is-active", active); button.setAttribute("aria-pressed", String(active)); });
    const filtered = state.filter === "all" ? state.history : state.history.filter(item => item.category === state.filter);
    if (state.historyError) { nodes.historyMessage.textContent = state.historyError; nodes.historyMessage.hidden = false; }
    else if (!state.storageAvailable) { nodes.historyMessage.textContent = "History is unavailable in this browser, but your current result is still visible."; nodes.historyMessage.hidden = false; }
    else if (!filtered.length) { nodes.historyMessage.textContent = state.history.length ? "No saved results match this filter." : "No analyses saved yet. Your recent results will appear here."; nodes.historyMessage.hidden = false; }
    else nodes.historyMessage.hidden = true;
    filtered.forEach(item => nodes.historyList.append(historyCard(item)));
  }
  function openHistoryItem(id) { const result = state.history.find(item => item.id === id); if (!result) return; clearResultStorageMessage(); setView("analyze"); renderResult(result, result.thumbnail); nodes.resultTitle.focus(); }
  function deleteHistoryItem(id) {
    const candidateHistory = state.history.filter(item => item.id !== id);
    const outcome = core.commitHistoryMutation(state.history, candidateHistory, history => localStorage.setItem(core.STORAGE_KEY, JSON.stringify(history)));
    if (!outcome.saved) { const failureMessage = "History item could not be deleted. Your saved history is unchanged."; state.storageAvailable = false; state.historyError = failureMessage; renderHistory(); nodes.historyMessage.focus(); announce(failureMessage); return; }
    state.history = outcome.history; state.storageAvailable = true; state.historyError = ""; renderHistory();
    const remainingAction = nodes.historyList.querySelector("[data-open-id], [data-delete-id]"); if (remainingAction) remainingAction.focus(); else nodes.historyMessage.focus(); announce("History item deleted.");
  }
  function clearHistory() {
    if (!window.confirm("Clear all saved detection history from this device?")) return;
    const candidateHistory = [];
    const outcome = core.commitHistoryMutation(state.history, candidateHistory, () => localStorage.removeItem(core.STORAGE_KEY));
    if (!outcome.saved) { const failureMessage = "History could not be cleared. Your saved history is unchanged."; state.storageAvailable = false; state.historyError = failureMessage; renderHistory(); announce(failureMessage); return; }
    state.history = outcome.history; state.storageAvailable = true; state.historyError = ""; renderHistory(); nodes.historyMessage.focus(); announce("Detection history cleared.");
  }
  async function analyzeCurrentFile() {
    if (!state.file || nodes.analyzeButton.disabled) return;
    nodes.analyzeButton.disabled = true; setUploadError(""); nodes.uploadPanel.hidden = true; nodes.progressPanel.hidden = false; announce("Analyzing image.");
    let failureMessage = GENERIC_ANALYSIS_ERROR;
    try {
      const formData = new FormData();
      formData.append("image", state.file);
      const response = await fetch("/api/analyze", {method: "POST", body: formData});
      let payload = null;
      try { payload = await response.json(); } catch (_) { payload = null; }
      if (!response.ok) {
        if (payload && typeof payload.error === "string" && payload.error.trim() && payload.error.length <= 300) failureMessage = payload.error.trim();
        throw new Error("request");
      }
      if (!payload || !core.validateServerResult(payload.result)) throw new Error("contract");
      const result = {...payload.result, thumbnail: state.thumbnail};
      const storageMessage = saveHistory(result); renderResult(result, state.imageUrl); announce(`${copyFor(result.category).title} Analysis complete.${storageMessage ? ` ${storageMessage}` : ""}`); nodes.resultTitle.focus();
    } catch (_) {
      nodes.uploadPanel.hidden = false; nodes.previewPanel.hidden = false; nodes.progressPanel.hidden = true; nodes.analyzeButton.disabled = false;
      setUploadError(failureMessage); announce(`Analysis failed. ${failureMessage}`); nodes.analyzeButton.focus();
    }
  }

  nodes.fileInput.addEventListener("change", event => { if (event.target.files[0]) selectFile(event.target.files[0]); });
  ["dragenter", "dragover"].forEach(type => nodes.dropZone.addEventListener(type, event => { event.preventDefault(); nodes.dropZone.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach(type => nodes.dropZone.addEventListener(type, event => { event.preventDefault(); nodes.dropZone.classList.remove("is-dragging"); if (type === "drop" && event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]); }));
  nodes.replaceButton.addEventListener("click", () => nodes.fileInput.click()); nodes.analyzeButton.addEventListener("click", analyzeCurrentFile); nodes.analyzeAnother.addEventListener("click", resetAnalyze);
  document.querySelectorAll("[data-view]").forEach(control => control.addEventListener("click", event => { event.preventDefault(); setView(control.dataset.view); }));
  nodes.filters.addEventListener("click", event => { const button = event.target.closest("[data-filter]"); if (button) { state.filter = button.dataset.filter; renderHistory(); } });
  nodes.historyList.addEventListener("click", event => { const open = event.target.closest("[data-open-id]"); const remove = event.target.closest("[data-delete-id]"); if (open) openHistoryItem(open.dataset.openId); if (remove) deleteHistoryItem(remove.dataset.deleteId); });
  nodes.clearHistory.addEventListener("click", clearHistory); loadHistory(); renderHistory(); setView(location.hash === "#history" ? "history" : "analyze");
});
