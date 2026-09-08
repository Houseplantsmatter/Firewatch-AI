const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../static/js/core.js");

function validEntry(overrides = {}) {
  return {
    id: "result-1",
    filename: "scene.jpg",
    timestamp: "2026-07-21T20:00:00.000Z",
    category: "fire",
    confidence: 0.84,
    detectionCount: 1,
    box: {x: 0.2, y: 0.2, width: 0.3, height: 0.3},
    thumbnail: "data:image/jpeg;base64,/9j/2Q==",
    ...overrides,
  };
}

function validServerResult(overrides = {}) {
  const {thumbnail, ...result} = validEntry(overrides);
  return result;
}

test("accepts supported images at the 10 MB boundary", () => {
  assert.equal(core.validateFile({name: "scene.JPG", type: "image/jpeg", size: core.MAX_BYTES}), "");
});

test("rejects unsupported, mismatched, empty, and oversized files", () => {
  assert.match(core.validateFile({name: "scene.gif", type: "image/gif", size: 20}), /JPG, PNG, or WebP/);
  assert.match(core.validateFile({name: "scene.png", type: "image/jpeg", size: 20}), /does not match/);
  assert.match(core.validateFile({name: "scene.jpg", type: "image/jpeg", size: 0}), /empty/);
  assert.match(core.validateFile({name: "scene.jpg", type: "image/jpeg", size: core.MAX_BYTES + 1}), /10 MB/);
});

test("accepts strict fire, smoke, and clear server results", () => {
  assert.equal(core.validateServerResult(validServerResult()), true);
  assert.equal(core.validateServerResult(validServerResult({category: "smoke"})), true);
  assert.equal(core.validateServerResult(validServerResult({
    category: "clear",
    confidence: null,
    detectionCount: 0,
    box: null,
  })), true);
});

test("rejects malformed or over-permissive server results", () => {
  const withoutId = validServerResult();
  delete withoutId.id;
  const corrupt = [
    null,
    [],
    {...validServerResult(), thumbnail: "data:image/jpeg;base64,/9j/2Q=="},
    {...validServerResult(), extra: true},
    withoutId,
    validServerResult({id: "x".repeat(201)}),
    validServerResult({filename: "x".repeat(256)}),
    validServerResult({timestamp: "not-a-date"}),
    validServerResult({category: "unknown"}),
    validServerResult({confidence: Number.NaN}),
    validServerResult({confidence: 1.01}),
    validServerResult({detectionCount: 1.5}),
    validServerResult({category: "clear", confidence: 0.8, detectionCount: 0, box: null}),
    validServerResult({category: "clear", confidence: null, detectionCount: 1, box: null}),
    validServerResult({category: "clear", confidence: null, detectionCount: 0}),
    validServerResult({box: {x: 0.8, y: 0.2, width: 0.3, height: 0.3}}),
    validServerResult({box: {x: 0.2, y: 0.2, width: 0.3, height: 0.3, unexpected: true}}),
  ];
  corrupt.forEach(value => assert.equal(core.validateServerResult(value), false));
});

test("history is newest first, valid only, and capped at 12", () => {
  const entries = Array.from({length: 15}, (_, index) => validEntry({id: String(index), filename: `${index}.jpg`, timestamp: new Date(Date.UTC(2026, 0, index + 1)).toISOString(), category: index % 2 ? "fire" : "clear", confidence: index % 2 ? 0.8 : null, detectionCount: index % 2 ? 1 : 0, box: index % 2 ? {x: .2, y: .2, width: .3, height: .3} : null}));
  entries.push({bad: true});
  const normalized = core.normalizeHistory(entries);
  assert.equal(normalized.length, core.MAX_HISTORY);
  assert.equal(normalized[0].id, "14");
  assert.equal(normalized[normalized.length - 1].id, "3");
});

test("history rejects category-dependent confidence, count, and box mismatches", () => {
  const corrupt = [
    validEntry({confidence: null}),
    validEntry({detectionCount: 0}),
    validEntry({box: null}),
    validEntry({category: "clear", confidence: 0.84, detectionCount: 0, box: null}),
    validEntry({category: "clear", confidence: null, detectionCount: 1, box: null}),
    validEntry({category: "clear", confidence: null, detectionCount: 0}),
  ];
  assert.deepEqual(core.normalizeHistory(corrupt), []);
});

test("history rejects non-finite values and boxes outside normalized extents", () => {
  const corrupt = [
    validEntry({confidence: Number.NaN}),
    validEntry({confidence: Number.POSITIVE_INFINITY}),
    validEntry({confidence: 1.01}),
    validEntry({detectionCount: 1.5}),
    validEntry({box: {x: -0.01, y: 0.2, width: 0.3, height: 0.3}}),
    validEntry({box: {x: 0.8, y: 0.2, width: 0.3, height: 0.3}}),
    validEntry({box: {x: 0.2, y: 0.9, width: 0.3, height: 0.2}}),
    validEntry({box: {x: 0.2, y: 0.2, width: 0, height: 0.3}}),
    validEntry({box: {x: 0.2, y: 0.2, width: 0.3, height: 0.3, unexpected: true}}),
  ];
  assert.deepEqual(core.normalizeHistory(corrupt), []);
});

test("history rejects invalid timestamps and missing or unbounded strings", () => {
  const corrupt = [
    validEntry({timestamp: "not-a-date"}),
    validEntry({timestamp: "2026-07-21"}),
    validEntry({id: ""}),
    validEntry({filename: "   "}),
    validEntry({id: "x".repeat(201)}),
    validEntry({filename: "x".repeat(256)}),
  ];
  assert.deepEqual(core.normalizeHistory(corrupt), []);
});

test("history accepts only bounded JPEG base64 data thumbnails", () => {
  const corrupt = [
    validEntry({thumbnail: "https://example.com/private.jpg"}),
    validEntry({thumbnail: "data:image/png;base64,eA=="}),
    validEntry({thumbnail: "data:image/jpeg;base64,not base64"}),
    validEntry({thumbnail: "data:image/jpeg;base64,"}),
    validEntry({thumbnail: "data:image/jpeg;base64,eA=="}),
    validEntry({thumbnail: `data:image/jpeg;base64,${"A".repeat(500001)}`}),
  ];
  assert.deepEqual(core.normalizeHistory(corrupt), []);
  assert.equal(core.normalizeHistory([validEntry()]).length, 1);
});

test("failed setItem keeps current history and a later success commits the candidate", () => {
  const current = [validEntry({id: "current"})];
  const candidate = [validEntry({id: "candidate"})];
  let shouldFail = true;
  let stored = null;
  const storage = {
    setItem(_key, value) {
      if (shouldFail) throw new Error("full");
      stored = JSON.parse(value);
    },
  };
  const persist = history => storage.setItem(core.STORAGE_KEY, JSON.stringify(history));

  const failed = core.commitHistoryMutation(current, candidate, persist);
  assert.equal(failed.saved, false);
  assert.equal(failed.history, current);
  assert.equal(stored, null);

  shouldFail = false;
  const recovered = core.commitHistoryMutation(current, candidate, persist);
  assert.equal(recovered.saved, true);
  assert.equal(recovered.history, candidate);
  assert.deepEqual(stored, candidate);
});

test("failed removeItem keeps current history and a later success clears it", () => {
  const current = [validEntry({id: "current"})];
  const candidate = [];
  let shouldFail = true;
  let removed = false;
  const storage = {
    removeItem(_key) {
      if (shouldFail) throw new Error("blocked");
      removed = true;
    },
  };
  const persist = () => storage.removeItem(core.STORAGE_KEY);

  const failed = core.commitHistoryMutation(current, candidate, persist);
  assert.equal(failed.saved, false);
  assert.equal(failed.history, current);
  assert.equal(removed, false);

  shouldFail = false;
  const recovered = core.commitHistoryMutation(current, candidate, persist);
  assert.equal(recovered.saved, true);
  assert.equal(recovered.history, candidate);
  assert.equal(removed, true);
});
