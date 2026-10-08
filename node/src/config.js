/**
 * Configuration loading and validation (port of src/config.py).
 *
 * Reads the same YAML files as the Python benchmark. LPIPS is not part of the
 * Node.js port: ``metrics.lpips``, ``weights.lpips`` and ``score_floors.lpips``
 * are accepted so the existing configs load unchanged, then dropped, and the
 * remaining weights are renormalised to sum to 1.
 */

import fs from 'node:fs';
import { load as loadYaml } from 'js-yaml';
import { getLogger } from './util.js';

const logger = getLogger('config');

export const WEIGHT_TOLERANCE = 1e-6;
export const METRIC_NAMES = ['ssim', 'silhouette', 'edge'];
export const DROPPED_METRICS = ['lpips'];
export const CROP_MODES = ['none', 'center_crop', 'foreground_bbox'];
export const MASK_MODES = ['alpha', 'auto', 'background'];
export const RESAMPLE_MODES = ['lanczos', 'bicubic', 'bilinear'];
export const ALIGNMENT_METHODS = ['none', 'centroid', 'phase_correlation'];
export const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];

export class ConfigError extends Error {}

const isNumber = (v) => typeof v === 'number' && Number.isFinite(v);
const isInt = (v) => Number.isInteger(v);

export function defaultConfig() {
  return {
    input: { extensions: ['.png', '.jpg', '.jpeg'], skip_unmatched: false },
    preprocessing: {
      canvas_size: 512,
      background_color: [255, 255, 255],
      crop_mode: 'none',
      foreground_padding: 0.1,
      mask_mode: 'auto',
      alpha_threshold: 16,
      mask_background_threshold: 12,
      mask_min_foreground_fraction: 0.001,
      mask_max_foreground_fraction: 0.98,
      resample: 'lanczos',
      alignment: 'phase_correlation',
      alignment_max_shift: 0.5,
    },
    metrics: {
      ssim: { gaussian_weights: true, sigma: 1.5, win_size: 7 },
      edge: { canny_low: 100, canny_high: 200, blur_kernel: 3, max_distance: 20.0, min_edge_fraction: 0.0005 },
    },
    // Python default {lpips 0.40, ssim 0.30, silhouette 0.20, edge 0.10} without LPIPS, renormalised.
    weights: { ssim: 0.5, silhouette: 1 / 3, edge: 1 / 6 },
    score_floors: {},
    score_gamma: 1.0,
    view_power: 1.0,
    mesh_complexity: { weight: 0.0, free_log2: 1.0, zero_log2: 5.0, mode: 'symmetric', bonus_weight: 0.0, bonus_log2: 2.0 },
    rig: {
      weight: 0.0,
      bone_free_log2: 1.0,
      bone_zero_log2: 4.0,
      motion_free_log2: 1.0,
      motion_zero_log2: 4.0,
      skeleton_max_distance: 0.25,
      motion_samples: 8,
    },
    output: { save_preprocessed: true, save_comparisons: true, report_pairs_per_page: 6, log_level: 'INFO', group_separator: '_' },
  };
}

// Allowed keys per section; nested sections are listed in NESTED.
const NESTED = {
  config: ['input', 'preprocessing', 'metrics', 'weights', 'score_floors', 'score_gamma', 'view_power', 'mesh_complexity', 'rig', 'output'],
  metrics: ['ssim', 'lpips', 'edge'],
};

function merge(section, data, path) {
  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    throw new ConfigError(`${path} must be a mapping, got ${Array.isArray(data) ? 'list' : typeof data}`);
  }
  const allowed = Object.keys(section);
  const unknown = Object.keys(data).filter((k) => !allowed.includes(k)).sort();
  if (unknown.length) {
    throw new ConfigError(`${path}: unknown key(s) ${JSON.stringify(unknown)}; allowed: ${JSON.stringify([...allowed].sort())}`);
  }
  for (const [k, v] of Object.entries(data)) section[k] = v;
}

/** Build and validate a configuration object from a parsed YAML mapping. */
export function configFromDict(data) {
  const cfg = defaultConfig();
  data = data ?? {};
  if (typeof data !== 'object' || Array.isArray(data)) throw new ConfigError('config must be a mapping');
  const unknown = Object.keys(data).filter((k) => !NESTED.config.includes(k)).sort();
  if (unknown.length) {
    throw new ConfigError(`config: unknown key(s) ${JSON.stringify(unknown)}; allowed: ${JSON.stringify([...NESTED.config].sort())}`);
  }
  for (const key of ['input', 'preprocessing', 'mesh_complexity', 'rig', 'output']) {
    if (key in data) merge(cfg[key], data[key], `config.${key}`);
  }
  if ('metrics' in data) {
    const m = data.metrics;
    if (m === null || typeof m !== 'object') throw new ConfigError('config.metrics must be a mapping');
    const bad = Object.keys(m).filter((k) => !NESTED.metrics.includes(k)).sort();
    if (bad.length) throw new ConfigError(`config.metrics: unknown key(s) ${JSON.stringify(bad)}; allowed: ${JSON.stringify(NESTED.metrics)}`);
    if (m.ssim !== undefined) merge(cfg.metrics.ssim, m.ssim, 'config.metrics.ssim');
    if (m.edge !== undefined) merge(cfg.metrics.edge, m.edge, 'config.metrics.edge');
    // metrics.lpips (net / device) has no meaning here and is ignored.
  }
  if ('weights' in data) cfg.weights = data.weights;
  if ('score_floors' in data) cfg.score_floors = data.score_floors;
  if ('score_gamma' in data) cfg.score_gamma = data.score_gamma;
  if ('view_power' in data) cfg.view_power = data.view_power;
  validateConfig(cfg);
  return cfg;
}

/** Load a YAML config file; ``null`` returns the built-in defaults. */
export function loadConfig(path = null) {
  if (path === null || path === undefined) {
    const cfg = defaultConfig();
    validateConfig(cfg);
    return cfg;
  }
  if (!fs.existsSync(path) || !fs.statSync(path).isFile()) throw new ConfigError(`Config file not found: ${path}`);
  let data;
  try {
    data = loadYaml(fs.readFileSync(path, 'utf-8'));
  } catch (exc) {
    throw new ConfigError(`Failed to parse YAML config ${path}: ${exc.message}`);
  }
  logger.debug(`Loaded config from ${path}`);
  return configFromDict(data ?? {});
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
export function validateConfig(cfg) {
  validateInput(cfg.input);
  validatePreprocessing(cfg.preprocessing);
  validateMetrics(cfg.metrics);
  validateMeshComplexity(cfg.mesh_complexity);
  validateRig(cfg.rig);
  cfg.weights = validateWeights(cfg.weights);
  cfg.score_floors = validateScoreFloors(cfg.score_floors);
  if (!isNumber(cfg.score_gamma)) throw new ConfigError(`score_gamma must be a number, got ${JSON.stringify(cfg.score_gamma)}`);
  if (!(cfg.score_gamma >= 0.1 && cfg.score_gamma <= 5.0)) throw new ConfigError(`score_gamma must be in [0.1, 5], got ${cfg.score_gamma}`);
  if (!isNumber(cfg.view_power)) throw new ConfigError(`view_power must be a number, got ${JSON.stringify(cfg.view_power)}`);
  if (!(cfg.view_power >= 0.05 && cfg.view_power <= 1.0)) throw new ConfigError(`view_power must be in [0.05, 1], got ${cfg.view_power}`);
  validateOutput(cfg.output);
  return cfg;
}

function validateInput(c) {
  if (!Array.isArray(c.extensions) || c.extensions.length === 0) throw new ConfigError('input.extensions must not be empty');
  c.extensions = c.extensions.map((e) => (String(e).startsWith('.') ? String(e).toLowerCase() : '.' + String(e).toLowerCase()));
  if (typeof c.skip_unmatched !== 'boolean') throw new ConfigError('input.skip_unmatched must be a boolean');
}

function validatePreprocessing(c) {
  if (!ALIGNMENT_METHODS.includes(c.alignment)) {
    throw new ConfigError(`preprocessing.alignment must be one of ${JSON.stringify(ALIGNMENT_METHODS)}, got ${JSON.stringify(c.alignment)}`);
  }
  if (!(c.alignment_max_shift > 0 && c.alignment_max_shift <= 1)) throw new ConfigError('preprocessing.alignment_max_shift must be in (0, 1]');
  if (!isInt(c.canvas_size) || c.canvas_size < 16) throw new ConfigError('preprocessing.canvas_size must be an integer >= 16');
  const bg = c.background_color;
  if (!Array.isArray(bg) || bg.length !== 3 || bg.some((v) => !isInt(v) || v < 0 || v > 255)) {
    throw new ConfigError('preprocessing.background_color must be three integers in [0, 255]');
  }
  if (!CROP_MODES.includes(c.crop_mode)) {
    throw new ConfigError(`preprocessing.crop_mode must be one of ${JSON.stringify(CROP_MODES)}, got ${JSON.stringify(c.crop_mode)}`);
  }
  if (!(c.foreground_padding >= 0 && c.foreground_padding < 1)) throw new ConfigError('preprocessing.foreground_padding must be in [0, 1)');
  if (!MASK_MODES.includes(c.mask_mode)) {
    throw new ConfigError(`preprocessing.mask_mode must be one of ${JSON.stringify(MASK_MODES)}, got ${JSON.stringify(c.mask_mode)}`);
  }
  if (!(c.alpha_threshold >= 0 && c.alpha_threshold <= 255)) throw new ConfigError('preprocessing.alpha_threshold must be in [0, 255]');
  if (!(c.mask_background_threshold >= 0 && c.mask_background_threshold <= 255)) {
    throw new ConfigError('preprocessing.mask_background_threshold must be in [0, 255]');
  }
  if (!(c.mask_min_foreground_fraction >= 0 && c.mask_min_foreground_fraction < 1)) {
    throw new ConfigError('preprocessing.mask_min_foreground_fraction must be in [0, 1)');
  }
  if (!(c.mask_min_foreground_fraction < c.mask_max_foreground_fraction && c.mask_max_foreground_fraction <= 1)) {
    throw new ConfigError('preprocessing.mask_max_foreground_fraction must be in (min_fraction, 1]');
  }
  if (!RESAMPLE_MODES.includes(c.resample)) {
    throw new ConfigError(`preprocessing.resample must be one of ${JSON.stringify(RESAMPLE_MODES)}, got ${JSON.stringify(c.resample)}`);
  }
}

function validateMetrics(m) {
  if (!(m.ssim.sigma > 0)) throw new ConfigError('metrics.ssim.sigma must be > 0');
  if (!isInt(m.ssim.win_size) || m.ssim.win_size < 3 || m.ssim.win_size % 2 === 0) {
    throw new ConfigError('metrics.ssim.win_size must be an odd integer >= 3');
  }
  const e = m.edge;
  if (!(e.canny_low >= 0 && e.canny_low <= e.canny_high && e.canny_high <= 255 * 4)) {
    throw new ConfigError('metrics.edge: require 0 <= canny_low <= canny_high');
  }
  if (!isInt(e.blur_kernel) || e.blur_kernel < 0 || (e.blur_kernel > 0 && e.blur_kernel % 2 === 0)) {
    throw new ConfigError('metrics.edge.blur_kernel must be 0 or an odd integer');
  }
  if (!(e.max_distance > 0)) throw new ConfigError('metrics.edge.max_distance must be > 0');
  if (!(e.min_edge_fraction >= 0 && e.min_edge_fraction < 1)) throw new ConfigError('metrics.edge.min_edge_fraction must be in [0, 1)');
}

function validateOutput(o) {
  if (!(o.report_pairs_per_page >= 1)) throw new ConfigError('output.report_pairs_per_page must be >= 1');
  if (o.group_separator === null || o.group_separator === undefined) o.group_separator = '';
  if (typeof o.group_separator !== 'string') {
    throw new ConfigError('output.group_separator must be a string (empty string disables grouping)');
  }
  o.log_level = String(o.log_level).toUpperCase();
  if (!LOG_LEVELS.includes(o.log_level)) throw new ConfigError(`output.log_level must be one of ${JSON.stringify(LOG_LEVELS)}`);
}

function validateMeshComplexity(c) {
  for (const name of ['weight', 'free_log2', 'zero_log2', 'bonus_weight', 'bonus_log2']) {
    if (!isNumber(c[name])) throw new ConfigError(`mesh_complexity.${name} must be a number, got ${JSON.stringify(c[name])}`);
  }
  if (!['fewer_is_better', 'symmetric'].includes(c.mode)) {
    throw new ConfigError(`mesh_complexity.mode must be 'fewer_is_better' or 'symmetric', got ${JSON.stringify(c.mode)}`);
  }
  if (!(c.weight >= 0 && c.weight <= 1)) throw new ConfigError(`mesh_complexity.weight must be in [0, 1], got ${c.weight}`);
  if (!(c.bonus_weight >= 0 && c.bonus_weight <= 1)) throw new ConfigError(`mesh_complexity.bonus_weight must be in [0, 1], got ${c.bonus_weight}`);
  if (c.bonus_log2 < 0) throw new ConfigError('mesh_complexity.bonus_log2 must be >= 0');
  if (c.free_log2 < 0) throw new ConfigError(`mesh_complexity.free_log2 must be >= 0, got ${c.free_log2}`);
  if (c.zero_log2 <= c.free_log2) {
    throw new ConfigError(`mesh_complexity.zero_log2 (${c.zero_log2}) must be greater than free_log2 (${c.free_log2})`);
  }
}

function validateRig(c) {
  for (const name of ['weight', 'bone_free_log2', 'bone_zero_log2', 'motion_free_log2', 'motion_zero_log2', 'skeleton_max_distance']) {
    if (!isNumber(c[name])) throw new ConfigError(`rig.${name} must be a number, got ${JSON.stringify(c[name])}`);
  }
  if (!(c.weight >= 0 && c.weight <= 1)) throw new ConfigError(`rig.weight must be in [0, 1], got ${c.weight}`);
  for (const [free, zero] of [['bone_free_log2', 'bone_zero_log2'], ['motion_free_log2', 'motion_zero_log2']]) {
    if (c[free] < 0 || c[zero] <= c[free]) throw new ConfigError(`rig.${zero} must be greater than rig.${free} (>= 0)`);
  }
  if (c.skeleton_max_distance <= 0) throw new ConfigError(`rig.skeleton_max_distance must be > 0, got ${c.skeleton_max_distance}`);
  if (!isInt(c.motion_samples) || c.motion_samples < 1) {
    throw new ConfigError(`rig.motion_samples must be a positive integer, got ${JSON.stringify(c.motion_samples)}`);
  }
}

/** Per-metric floors in [0, 100); LPIPS floors are dropped. */
export function validateScoreFloors(floors) {
  floors = floors ?? {};
  if (typeof floors !== 'object' || Array.isArray(floors)) throw new ConfigError('score_floors must be a mapping of metric name -> floor score');
  const known = [...METRIC_NAMES, ...DROPPED_METRICS];
  const unknown = Object.keys(floors).filter((k) => !known.includes(k)).sort();
  if (unknown.length) throw new ConfigError(`score_floors: unknown metric(s) ${JSON.stringify(unknown)}; expected ${JSON.stringify(METRIC_NAMES)}`);
  const out = {};
  for (const name of METRIC_NAMES) {
    const value = floors[name] ?? 0.0;
    if (!isNumber(value)) throw new ConfigError(`score_floors.${name} must be a number, got ${JSON.stringify(value)}`);
    if (!(value >= 0 && value < 100)) throw new ConfigError(`score_floors.${name} must be in [0, 100), got ${value}`);
    out[name] = value;
  }
  return out;
}

/**
 * Validate weights like the Python benchmark (known names, all present, >= 0,
 * sum 1), then drop LPIPS and renormalise the rest. A config that already
 * has no ``lpips`` key must sum to 1 on its own.
 */
export function validateWeights(weights) {
  if (weights === null || typeof weights !== 'object' || Array.isArray(weights)) {
    throw new ConfigError('weights must be a mapping of metric name -> weight');
  }
  const known = [...METRIC_NAMES, ...DROPPED_METRICS];
  const unknown = Object.keys(weights).filter((k) => !known.includes(k)).sort();
  if (unknown.length) throw new ConfigError(`weights: unknown metric(s) ${JSON.stringify(unknown)}; expected ${JSON.stringify(METRIC_NAMES)}`);
  const missing = METRIC_NAMES.filter((k) => !(k in weights)).sort();
  if (missing.length) throw new ConfigError(`weights: missing metric(s) ${JSON.stringify(missing)}`);
  const all = {};
  for (const name of Object.keys(weights)) {
    const value = weights[name];
    if (typeof value !== 'number') throw new ConfigError(`weights.${name} must be a number, got ${JSON.stringify(value)}`);
    if (!Number.isFinite(value)) throw new ConfigError(`weights.${name} must be finite`);
    if (value < 0) throw new ConfigError(`weights.${name} must be >= 0, got ${value}`);
    all[name] = value;
  }
  const total = Object.values(all).reduce((a, b) => a + b, 0);
  if (Math.abs(total - 1.0) > WEIGHT_TOLERANCE) {
    throw new ConfigError(`weights must sum to 1.0 (tolerance ${WEIGHT_TOLERANCE}), got ${total.toFixed(6)}`);
  }
  const dropped = DROPPED_METRICS.filter((k) => (all[k] ?? 0) > 0);
  const kept = METRIC_NAMES.reduce((s, k) => s + all[k], 0);
  if (kept <= 0) throw new ConfigError('weights: every weight is on lpips, which the Node.js port does not compute');
  const out = {};
  for (const name of METRIC_NAMES) out[name] = dropped.length ? all[name] / kept : all[name];
  if (dropped.length) {
    logger.debug(
      `LPIPS is not computed in the Node.js port; dropped weight ${all.lpips} and renormalised to ` +
        METRIC_NAMES.map((k) => `${k} ${out[k].toFixed(4)}`).join(', '),
    );
  }
  return out;
}
