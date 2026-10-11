// Learner progress, shared with the course site: the same localStorage entry and shape as its
// src/lib/learnfsd-progress.ts, so a pass written here is read there (same origin).
export const STORAGE_KEY = "learnfsd.v2.progress"; // must equal the site's src/lib/learnfsd-progress.ts:7

const isRecord = (v) => typeof v === "object" && v !== null && !Array.isArray(v);

/** The stored progress object, or null when absent, malformed or unreadable. Never throws. */
export function readProgress() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    return isRecord(data) && data.version === 1 && isRecord(data.units) && isRecord(data.deck) ? data : null;
  } catch {
    return null;
  }
}

/** Record that `exerciseId` passed in unit `unitKey`; the first pass keeps its timestamp. Returns false when storage refuses. */
export function markPassed(unitKey, exerciseId, now = new Date()) {
  try {
    const progress = readProgress() ?? { version: 1, units: {}, deck: {} };
    const unit = { logs: [], ...progress.units[unitKey] }; // `logs` is required by the site's addPilotLog
    const passed = unit.exercises?.[exerciseId] ?? { passedAt: now.toISOString(), via: "playground" };
    progress.units[unitKey] = { ...unit, exercises: { ...unit.exercises, [exerciseId]: passed } };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
    return true;
  } catch {
    return false;
  }
}
