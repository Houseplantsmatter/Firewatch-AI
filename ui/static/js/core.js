(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.FireWatchCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const MAX_BYTES = 10 * 1024 * 1024;
  const MAX_HISTORY = 12;
  const MAX_THUMBNAIL_LENGTH = 500000;
  const STORAGE_KEY = "firewatch-ai-history-v1";
  const TYPES = {"image/jpeg": ["jpg", "jpeg"], "image/png": ["png"], "image/webp": ["webp"]};
  const SERVER_RESULT_KEYS = ["box", "category", "confidence", "detectionCount", "filename", "id", "timestamp"];
  const HISTORY_RESULT_KEYS = [...SERVER_RESULT_KEYS, "thumbnail"].sort();

  function validateFile(file) {
    if (!file || typeof file.name !== "string") return "Choose an image to continue.";
    if (!Number.isFinite(file.size) || file.size <= 0) return "This image is empty. Choose a different file.";
    if (file.size > MAX_BYTES) return "This image is larger than 10 MB. Choose a smaller file.";
    const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!Object.prototype.hasOwnProperty.call(TYPES, file.type)) return "Choose a JPG, PNG, or WebP image.";
    if (!TYPES[file.type].includes(extension)) return "The file extension does not match the image type.";
    return "";
  }

  function isBoundedString(value, maximumLength) {
    return typeof value === "string" && value.trim().length > 0 && value.length <= maximumLength;
  }

  function isValidTimestamp(value) {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) return false;
    const parsed = new Date(value);
    return Number.isFinite(parsed.getTime()) && parsed.toISOString() === value;
  }

  function isValidThumbnail(value) {
    if (typeof value !== "string" || value.length > MAX_THUMBNAIL_LENGTH) return false;
    return /^data:image\/jpeg;base64,(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value) &&
      value.startsWith("data:image/jpeg;base64,/9j/");
  }

  function isValidBox(box) {
    if (!box || typeof box !== "object" || Array.isArray(box)) return false;
    const {x, y, width, height} = box;
    return hasExactKeys(box, ["height", "width", "x", "y"]) &&
      [x, y, width, height].every(Number.isFinite) && x >= 0 && y >= 0 && width > 0 && height > 0 &&
      x + width <= 1 && y + height <= 1;
  }

  function hasExactKeys(value, expectedKeys) {
    const keys = Object.keys(value).sort();
    return keys.length === expectedKeys.length && keys.every((key, index) => key === expectedKeys[index]);
  }

  function hasValidResultFields(entry) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry) ||
        !isBoundedString(entry.id, 200) || !isBoundedString(entry.filename, 255) ||
        !isValidTimestamp(entry.timestamp) || !["fire", "smoke", "clear"].includes(entry.category) ||
        !Number.isInteger(entry.detectionCount) || entry.detectionCount < 0) return false;
    if (entry.category === "clear") {
      return entry.confidence === null && entry.detectionCount === 0 && entry.box === null;
    }
    return Number.isFinite(entry.confidence) && entry.confidence >= 0 && entry.confidence <= 1 &&
      entry.detectionCount > 0 && isValidBox(entry.box);
  }

  function validateServerResult(entry) {
    return !!entry && typeof entry === "object" && !Array.isArray(entry) &&
      hasExactKeys(entry, SERVER_RESULT_KEYS) && hasValidResultFields(entry);
  }

  function isValidEntry(entry) {
    return !!entry && typeof entry === "object" && !Array.isArray(entry) &&
      hasExactKeys(entry, HISTORY_RESULT_KEYS) && hasValidResultFields(entry) &&
      isValidThumbnail(entry.thumbnail);
  }

  function normalizeHistory(value) {
    if (!Array.isArray(value)) return [];
    return value.filter(isValidEntry).sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)).slice(0, MAX_HISTORY);
  }

  function commitHistoryMutation(currentHistory, candidateHistory, persist) {
    try {
      persist(candidateHistory);
      return {history: candidateHistory, saved: true};
    } catch (_) {
      return {history: currentHistory, saved: false};
    }
  }

  return {MAX_BYTES, MAX_HISTORY, STORAGE_KEY, validateFile, validateServerResult, normalizeHistory, commitHistoryMutation};
});
