/**
 * Command-line interface (port of the scoring commands of src/cli.py).
 *
 *     imgsim compare --reference data/reference --candidate data/candidate
 *     imgsim compare-pair --reference a/front.png --candidate b/front.png
 *     imgsim render-views --model models/mug.glb --output renders/mug
 *     imgsim compare-models --reference models/a.glb --candidate models/b.glb
 *     imgsim rig-info --model models/character.glb
 *
 * Models must be local files (glb / gltf / obj / stl); Sketchfab download and
 * AI generation stay in the Python CLI.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BenchmarkResult, BenchmarkRunner, computeOverallScore, PairingError, slugify } from './benchmark.js';
import { meshComplexity } from './complexity.js';
import { ALIGNMENT_METHODS, ConfigError, CROP_MODES, loadConfig, validateConfig } from './config.js';
import { applyOrientation, autoOrient, orientResultToDict } from './orient.js';
import {
  AXIS_NAMES,
  canonicalRotation,
  DEFAULT_VIEWS,
  defaultRenderOptions,
  loadMesh,
  matMul3,
  parseViews,
  RenderError,
  renderViews,
  STYLES,
  validateRenderOptions,
  VIEWS,
  yawMatrix,
} from './render.js';
import { formatSummaryTable, saveMetricsCsv, saveMetricsJson, saveReport } from './reporting.js';
import { analyseRig, rigComparison, rigInfoToDict, summarizeRig } from './rig.js';
import { getLogger, setLogLevel } from './util.js';

export const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DEFAULT_CONFIG = path.join(PROJECT_ROOT, 'configs', 'default.yaml');
// compare-models defaults to the shape-first config (crop to the object, silhouette + edges)
const SHAPE_CONFIG = path.join(PROJECT_ROOT, 'configs', 'shape.yaml');
const DEFAULT_OUTPUT = path.join(PROJECT_ROOT, 'outputs');

const logger = getLogger('cli');

class UsageError extends Error {}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
const COMMON = {
  reference: { type: 'string', required: true },
  candidate: { type: 'string', required: true },
  output: { type: 'string', default: DEFAULT_OUTPUT },
  config: { type: 'string' },
  'crop-mode': { type: 'string', choices: CROP_MODES },
  alignment: { type: 'string', choices: ALIGNMENT_METHODS },
  'canvas-size': { type: 'int' },
  'no-save': { type: 'boolean' },
  'log-level': { type: 'string' },
};
const RENDER = {
  views: { type: 'string', default: DEFAULT_VIEWS.join(',') },
  size: { type: 'int', default: 512 },
  up: { type: 'string', choices: AXIS_NAMES, default: '+y' },
  front: { type: 'string', choices: AXIS_NAMES },
  fill: { type: 'float', default: 0.85 },
  style: { type: 'string', choices: STYLES, default: 'shaded' },
  supersample: { type: 'int', default: 2 },
};
const MODEL_BENCH = {
  reference: { type: 'string', required: true },
  candidate: { type: 'string', required: true },
  output: { type: 'string', default: DEFAULT_OUTPUT },
  config: { type: 'string' },
  'mesh-weight': { type: 'float' },
  'rig-weight': { type: 'float' },
  'crop-mode': { type: 'string', choices: CROP_MODES },
  alignment: { type: 'string', choices: ALIGNMENT_METHODS },
  'canvas-size': { type: 'int' },
  'no-save': { type: 'boolean' },
  'log-level': { type: 'string' },
  label: { type: 'string' },
  'auto-orient': { type: 'boolean', negatable: true, default: false },
  'candidate-up': { type: 'string', choices: AXIS_NAMES },
  'candidate-front': { type: 'string', choices: AXIS_NAMES },
  'candidate-yaw': { type: 'float', default: 0.0 },
};

const COMMANDS = {
  compare: { help: 'Compare two folders of images paired by file name', spec: { ...COMMON, 'skip-unmatched': { type: 'boolean' } } },
  'compare-pair': { help: 'Compare a single reference/candidate image pair', spec: COMMON },
  'render-views': {
    help: 'Render orthographic views (front/side/top ...) of a 3D model to PNG',
    spec: { model: { type: 'string', required: true }, output: { type: 'string', required: true }, ...RENDER, 'log-level': { type: 'string', default: 'INFO' } },
  },
  'compare-models': { help: 'Render two 3D models with identical settings and score their views', spec: { ...MODEL_BENCH, ...RENDER } },
  'rig-info': {
    help: 'Print skeleton / skinning / animation facts of a glTF or GLB model as JSON',
    spec: { model: { type: 'string', required: true }, 'motion-samples': { type: 'int', default: 8 }, 'log-level': { type: 'string', default: 'INFO' } },
  },
};

function usage() {
  const lines = ['Usage: imgsim <command> [options]', '', 'Image similarity benchmark (SSIM, silhouette IoU, edge similarity; no LPIPS).', '', 'Commands:'];
  for (const [name, c] of Object.entries(COMMANDS)) lines.push(`  ${name.padEnd(16)} ${c.help}`);
  lines.push('', 'Run "imgsim <command> --help" for the options of a command.');
  return lines.join('\n');
}

function commandUsage(name) {
  const lines = [`Usage: imgsim ${name} [options]`, '', COMMANDS[name].help, '', 'Options:'];
  for (const [opt, s] of Object.entries(COMMANDS[name].spec)) {
    let text = s.type === 'boolean' ? `--${opt}` : `--${opt} <${s.type === 'string' ? 'value' : s.type}>`;
    if (s.negatable) text += ` / --no-${opt}`;
    const notes = [];
    if (s.required) notes.push('required');
    if (s.choices) notes.push(s.choices.join('|'));
    if (s.default !== undefined && s.type !== 'boolean') notes.push(`default: ${s.default}`);
    lines.push(`  ${text.padEnd(34)} ${notes.join('; ')}`);
  }
  return lines.join('\n');
}

export function parseArgs(argv) {
  const [command, ...rest] = argv;
  if (!command || command === '--help' || command === '-h') return { command: null, help: true };
  if (!(command in COMMANDS)) throw new UsageError(`unknown command '${command}'\n\n${usage()}`);
  const spec = COMMANDS[command].spec;
  const args = {};
  for (const [k, s] of Object.entries(spec)) if (s.default !== undefined) args[k] = s.default;
  for (let i = 0; i < rest.length; i++) {
    const tok = rest[i];
    if (tok === '--help' || tok === '-h') return { command, help: true };
    if (!tok.startsWith('--')) throw new UsageError(`unexpected argument '${tok}'`);
    let key = tok.slice(2);
    let value;
    if (key.includes('=')) [key, value] = [key.slice(0, key.indexOf('=')), key.slice(key.indexOf('=') + 1)];
    if (!(key in spec) && key.startsWith('no-') && spec[key.slice(3)]?.negatable) {
      args[key.slice(3)] = false;
      continue;
    }
    const s = spec[key];
    if (!s) throw new UsageError(`unknown option --${key} for ${command}\n\n${commandUsage(command)}`);
    if (s.type === 'boolean') {
      args[key] = true;
      continue;
    }
    if (value === undefined) {
      value = rest[++i];
      if (value === undefined) throw new UsageError(`option --${key} needs a value`);
    }
    if (s.type === 'int' || s.type === 'float') {
      const num = Number(value);
      if (!Number.isFinite(num) || (s.type === 'int' && !Number.isInteger(num))) throw new UsageError(`option --${key}: invalid ${s.type} '${value}'`);
      value = num;
    }
    if (s.choices && !s.choices.includes(value)) throw new UsageError(`option --${key}: invalid choice '${value}' (choose from ${s.choices.join(', ')})`);
    args[key] = value;
  }
  for (const [k, s] of Object.entries(spec)) if (s.required && args[k] === undefined) throw new UsageError(`missing required option --${k}\n\n${commandUsage(command)}`);
  return { command, args };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
/** Load ``--config`` (or ``fallback``) and apply the CLI overrides. */
function loadEffectiveConfig(args, fallback = DEFAULT_CONFIG) {
  let configPath = args.config ?? null;
  if (configPath === null && fs.existsSync(fallback)) configPath = fallback;
  const cfg = loadConfig(configPath);
  if (args['crop-mode'] !== undefined) cfg.preprocessing.crop_mode = args['crop-mode'];
  if (args.alignment !== undefined) cfg.preprocessing.alignment = args.alignment;
  if (args['canvas-size'] !== undefined) cfg.preprocessing.canvas_size = args['canvas-size'];
  if (args['skip-unmatched']) cfg.input.skip_unmatched = true;
  if (args['mesh-weight'] !== undefined) cfg.mesh_complexity.weight = args['mesh-weight'];
  if (args['rig-weight'] !== undefined) cfg.rig.weight = args['rig-weight'];
  if (args['log-level'] !== undefined) cfg.output.log_level = args['log-level'];
  validateConfig(cfg);
  return cfg;
}

function renderOptionsFromArgs(args) {
  const front = args.front ?? { '+y': '+z', '-y': '-z', '+z': '-y', '-z': '+y', '+x': '+z', '-x': '+z' }[args.up];
  const opts = {
    ...defaultRenderOptions(),
    size: args.size,
    views: parseViews(args.views),
    up: args.up,
    front,
    fill: args.fill,
    style: args.style,
    supersample: args.supersample,
  };
  validateRenderOptions(opts);
  return opts;
}

function resolveModel(ref) {
  const p = ref.startsWith('~') ? path.join(os.homedir(), ref.slice(1)) : ref;
  if (fs.existsSync(p) && fs.statSync(p).isFile()) return p;
  throw new RenderError(`model not found: '${ref}' (the Node.js port takes local files only; download Sketchfab models with the Python CLI)`);
}

/** Short model name for run folders: generated-model descriptor or Sketchfab name from the sidecar JSON, else the file stem. */
export function describeModel(file, generation = null) {
  if (generation) {
    const meta = generation.meta ?? {};
    const ver = String(meta.model_version_used || meta.model_version || '').replace(/-\d{8}$/, '');
    return [generation.provider, generation.mode, ver].filter(Boolean).join('-');
  }
  const sidecar = path.join(path.dirname(file), path.parse(file).name + '.json');
  if (fs.existsSync(sidecar)) {
    try {
      const data = JSON.parse(fs.readFileSync(sidecar, 'utf-8'));
      if (data.name && data.uid) return String(data.name);
      if (data.provider) return describeModel(file, data);
    } catch {
      // unreadable sidecar: fall back to the file name
    }
  }
  return path.parse(file).name;
}

const pairLabel = (ref, cand, sideLen = 36) => `${slugify(ref, sideLen)}_vs_${slugify(cand, sideLen)}`;

function printResult(result) {
  console.log();
  console.log(formatSummaryTable(result));
  if (result.run_dir !== null) console.log(`\nOutputs written to: ${result.run_dir}`);
  return result.overall_score !== null ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
function cmdCompare(args) {
  const cfg = loadEffectiveConfig(args);
  setLogLevel(cfg.output.log_level);
  const runner = new BenchmarkRunner(cfg);
  const result = runner.run(args.reference, args.candidate, args['no-save'] ? null : args.output);
  return printResult(result);
}

function cmdComparePair(args) {
  const cfg = loadEffectiveConfig(args);
  setLogLevel(cfg.output.log_level);
  for (const [label, p] of [['Reference', args.reference], ['Candidate', args.candidate]]) {
    if (!fs.existsSync(p) || !fs.statSync(p).isFile()) throw new PairingError(`${label} image not found: ${p}`);
  }
  const runner = new BenchmarkRunner(cfg);
  const runDir = args['no-save'] ? null : BenchmarkRunner.createRunDir(args.output);
  const name = path.basename(args.reference);
  if (name.toLowerCase() !== path.basename(args.candidate).toLowerCase()) {
    logger.warning(`File names differ (${name} vs ${path.basename(args.candidate)}); comparing anyway because compare-pair was requested explicitly`);
  }
  const pair = runner.comparePair(args.reference, args.candidate, name, runDir);
  const result = new BenchmarkResult({ pairs: [pair], overallScore: computeOverallScore([pair]), runDir, config: cfg });
  if (runDir !== null) {
    saveMetricsJson(result, path.join(runDir, 'metrics.json'));
    saveMetricsCsv(result, path.join(runDir, 'metrics.csv'));
    saveReport(result, runDir);
  }
  console.log();
  console.log(JSON.stringify({ pair: pair.toDict(), overall_score: result.overall_score }, null, 2));
  if (runDir !== null) console.log(`\nOutputs written to: ${runDir}`);
  return pair.ok ? 0 : 1;
}

function cmdRenderViews(args) {
  setLogLevel(args['log-level']);
  const opts = renderOptionsFromArgs(args);
  const written = renderViews(resolveModel(args.model), args.output, opts);
  console.log();
  for (const [view, p] of Object.entries(written)) console.log(`${view.padStart(8)}: ${p}`);
  return 0;
}

/**
 * Render both models (orienting the candidate if asked) and score them. With
 * auto-orient, when the silhouette search picked a yaw other than 0 the
 * axis-aligned orientation is scored as well and the higher overall score wins.
 */
export function runModelComparison(cfg, opts, refPath, candPath, runDir, { auto = false, candUp = null, candFront = null, candYaw = 0.0, extraMeta = null } = {}) {
  const refMesh = loadMesh(refPath, opts.up, opts.front);
  const base = loadMesh(candPath);
  let orientInfo = null;
  let candidates;
  if (auto) {
    const best = autoOrient(base, refMesh, opts.views);
    orientInfo = orientResultToDict(best);
    candidates = [[best.up, best.front, best.yaw]];
    if (Math.abs(best.yaw) > 1e-6) candidates.push([best.up, best.front, 0.0]);
    logger.info(`Candidate orientation from silhouettes: up=${best.up} front=${best.front} yaw=${best.yaw.toFixed(1)}`);
  } else {
    candidates = [[candUp ?? opts.up, candFront ?? opts.front, candYaw]];
  }

  const runner = new BenchmarkRunner(cfg);
  const refRig = analyseRig(refPath, cfg.rig.motion_samples);
  const candRig = analyseRig(candPath, cfg.rig.motion_samples);

  const score = (up, front, yaw, outDir) => {
    const candMesh = applyOrientation(base, up, front, yaw);
    const tmp = outDir === null ? fs.mkdtempSync(path.join(os.tmpdir(), 'imgsim_render_')) : null;
    try {
      const renderRoot = tmp ?? path.join(outDir, 'renders');
      const refDir = path.join(renderRoot, 'reference');
      const candDir = path.join(renderRoot, 'candidate');
      logger.info(`Rendering reference model ${refPath}`);
      renderViews(refMesh, refDir, opts);
      logger.info(`Rendering candidate model ${candPath} (up=${up} front=${front} yaw=${yaw.toFixed(1)})`);
      renderViews(candMesh, candDir, opts);
      const meshInfo = meshComplexity(
        { faces: refMesh.numFaces, vertices: refMesh.numVertices },
        { faces: candMesh.numFaces, vertices: candMesh.numVertices },
        cfg.mesh_complexity,
      );
      // compare the skeletons in the same orientation as the renders
      const rigInfo = rigComparison(refRig, candRig, cfg.rig, canonicalRotation(opts.up, opts.front), matMul3(yawMatrix(yaw), canonicalRotation(up, front)));
      return runner.run(refDir, candDir, null, { runDir: outDir, meshComplexity: meshInfo, rig: rigInfo });
    } finally {
      if (tmp !== null) fs.rmSync(tmp, { recursive: true, force: true });
    }
  };

  const results = [score(...candidates[0], runDir)];
  for (const c of candidates.slice(1)) results.push(score(...c, null));
  const scores = results.map((r) => (r.overall_score !== null ? r.overall_score : -1.0));
  let winner = 0;
  scores.forEach((s, i) => {
    if (s > scores[winner]) winner = i;
  });
  let result;
  if (winner !== 0) {
    logger.info(`Axis-aligned orientation (yaw 0) scores ${scores[winner].toFixed(2)} vs ${scores[0].toFixed(2)} with yaw ${candidates[0][2].toFixed(1)}; keeping yaw 0`);
    result = runDir !== null ? score(...candidates[winner], runDir) : results[winner];
  } else {
    result = results[0];
  }
  const chosen = candidates[winner];
  if (orientInfo !== null) {
    orientInfo.yaw = chosen[2];
    orientInfo.scored = candidates.map((c, i) => ({ up: c[0], front: c[1], yaw: c[2], overall_score: scores[i] }));
  }
  if (runDir !== null) {
    const modelsMeta = {
      reference: { model: refPath, up: opts.up, front: opts.front },
      candidate: { model: candPath, up: chosen[0], front: chosen[1], yaw: chosen[2], auto_orient: orientInfo },
      render: { ...opts },
      ...(extraMeta ?? {}),
    };
    fs.writeFileSync(path.join(runDir, 'models.json'), JSON.stringify(modelsMeta, null, 2), 'utf-8');
  }
  return result;
}

function cmdCompareModels(args) {
  const cfg = loadEffectiveConfig(args, SHAPE_CONFIG);
  setLogLevel(cfg.output.log_level);
  const opts = renderOptionsFromArgs(args);
  const refPath = resolveModel(args.reference);
  const candPath = resolveModel(args.candidate);
  const label = args.label ?? pairLabel(describeModel(refPath), describeModel(candPath));
  const runDir = args['no-save'] ? null : BenchmarkRunner.createRunDir(args.output, label);
  const result = runModelComparison(cfg, opts, refPath, candPath, runDir, {
    auto: args['auto-orient'],
    candUp: args['candidate-up'] ?? null,
    candFront: args['candidate-front'] ?? null,
    candYaw: args['candidate-yaw'],
  });
  return printResult(result);
}

function cmdRigInfo(args) {
  setLogLevel(args['log-level']);
  const info = analyseRig(args.model, args['motion-samples']);
  console.log(JSON.stringify(rigInfoToDict(info), null, 2));
  process.stderr.write(`\n${path.basename(args.model)}: ${summarizeRig(info)}\n`);
  return info.readable ? 0 : 1;
}

const HANDLERS = {
  compare: cmdCompare,
  'compare-pair': cmdComparePair,
  'render-views': cmdRenderViews,
  'compare-models': cmdCompareModels,
  'rig-info': cmdRigInfo,
};

export function main(argv = process.argv.slice(2)) {
  let parsed;
  try {
    parsed = parseArgs(argv);
  } catch (exc) {
    if (exc instanceof UsageError) {
      process.stderr.write(`Error: ${exc.message}\n`);
      return 2;
    }
    throw exc;
  }
  if (parsed.help) {
    console.log(parsed.command ? commandUsage(parsed.command) : usage());
    return 0;
  }
  try {
    return HANDLERS[parsed.command](parsed.args);
  } catch (exc) {
    if (exc instanceof ConfigError) {
      process.stderr.write(`Configuration error: ${exc.message}\n`);
      return 2;
    }
    if (exc instanceof PairingError) {
      process.stderr.write(`Error: ${exc.message}\n`);
      return 3;
    }
    if (exc instanceof RenderError) {
      process.stderr.write(`Error: ${exc.message}\n`);
      return 4;
    }
    throw exc;
  }
}
