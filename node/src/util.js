/**
 * Small shared helpers: logging and the rounding rules of numpy / Python.
 */

const LEVELS = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40 };
let currentLevel = LEVELS.INFO;

export function setLogLevel(level) {
  const key = String(level || 'INFO').toUpperCase();
  currentLevel = LEVELS[key] ?? LEVELS.INFO;
}

function stamp() {
  const d = new Date();
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((v) => String(v).padStart(2, '0')).join(':');
}

/** Logger writing ``HH:MM:SS LEVEL name: message`` lines to stderr, like the Python CLI. */
export function getLogger(name) {
  const emit = (level, msg) => {
    if (LEVELS[level] < currentLevel) return;
    process.stderr.write(`${stamp()} ${level.padEnd(7)} ${name}: ${msg}\n`);
  };
  return {
    debug: (msg) => emit('DEBUG', msg),
    info: (msg) => emit('INFO', msg),
    warning: (msg) => emit('WARNING', msg),
    error: (msg) => emit('ERROR', msg),
  };
}

/** Round half to even, like ``np.rint`` / ``np.round`` / Python's ``round``. */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

/** ``%.Nf`` formatting with ``n/a`` for missing values (reporting._fmt). */
export function fmt(value, digits = 2, na = 'n/a') {
  if (value === null || value === undefined || Number.isNaN(value)) return na;
  return Number(value).toFixed(digits);
}

export function mean(values) {
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}
