/**
 * Benchmark orchestration (port of src/benchmark.py): discover, pair,
 * preprocess, align, score and save.
 *
 *     outputs/run_YYYYMMDD_HHMMSS[_label]/
 *         preprocessed/{reference,candidate}/*.png
 *         comparisons/*.png
 *         metrics.json
 *         metrics.csv
 *         report.html
 */

import fs from 'node:fs';
import path from 'node:path';
import * as reporting from './reporting.js';
import { applyPenalties } from './complexity.js';
import { alignCandidate } from './alignment.js';
import { METRIC_NAMES } from './config.js';
import {
  applyScoreFloor,
  combineScores,
  computeEdgeSimilarity,
  computeSilhouetteIou,
  computeSsim,
  iouToScore,
  ssimToScore,
} from './metrics.js';
import { ImageLoadError, preprocessImage, savePreprocessed } from './preprocessing.js';
import { fmt, getLogger, mean } from './util.js';

const logger = getLogger('benchmark');

export class PairingError extends Error {}

/** All metrics for one image pair; ``null`` means not available. */
export class PairResult {
  constructor(name, referencePath, candidatePath) {
    this.name = name;
    this.reference_path = referencePath;
    this.candidate_path = candidatePath;
    this.ssim = null;
    this.ssim_score = null;
    this.silhouette_iou = null;
    this.silhouette_score = null;
    this.edge_score = null;
    this.edge_chamfer_ref_to_cand = null;
    this.edge_chamfer_cand_to_ref = null;
    this.pair_score = null;
    this.calibrated_scores = {};
    this.effective_weights = {};
    this.unavailable_metrics = [];
    this.mask_source_reference = 'none';
    this.mask_source_candidate = 'none';
    this.alignment = null;
    this.error = null;
    this.preprocessed_reference = null;
    this.preprocessed_candidate = null;
    this.comparison_image = null;
    // in memory only
    this.refImage = null;
    this.candImage = null;
  }

  get ok() {
    return this.error === null && this.pair_score !== null;
  }

  toDict() {
    const { name, refImage, candImage, ...rest } = this;
    return rest;
  }
}

/** Result of a full run. */
export class BenchmarkResult {
  constructor({ pairs, overallScore, runDir, config, skippedUnmatched = {}, groupScores = {}, shapeScore = null, mesh = null, rig = null }) {
    this.pairs = pairs;
    this.overall_score = overallScore;
    this.run_dir = runDir;
    this.config = config;
    this.skipped_unmatched = skippedUnmatched;
    this.group_scores = groupScores;
    this.shape_score = shapeScore;
    this.mesh = mesh;
    this.rig = rig;
  }

  get validPairs() {
    return this.pairs.filter((p) => p.ok);
  }

  toDict() {
    return {
      pairs: Object.fromEntries(this.pairs.map((p) => [p.name, p.toDict()])),
      overall_score: this.overall_score,
      shape_score: this.shape_score,
      mesh: this.mesh,
      rig: this.rig,
      group_scores: this.group_scores,
      num_pairs: this.pairs.length,
      num_valid_pairs: this.validPairs.length,
      skipped_unmatched: this.skipped_unmatched,
      run_dir: this.run_dir,
      configuration: this.config,
    };
  }
}

// ---------------------------------------------------------------------------
// Discovery and pairing
// ---------------------------------------------------------------------------
/** ``Map(lower-case file name -> path)`` of supported images in ``folder``. */
export function discoverImages(folder, extensions) {
  if (!fs.existsSync(folder) || !fs.statSync(folder).isDirectory()) {
    throw new PairingError(`Input folder does not exist or is not a directory: ${folder}`);
  }
  const exts = new Set(extensions.map((e) => e.toLowerCase()));
  const found = new Map();
  for (const name of fs.readdirSync(folder).sort()) {
    const full = path.join(folder, name);
    if (!fs.statSync(full).isFile() || !exts.has(path.extname(name).toLowerCase())) continue;
    const key = name.toLowerCase();
    if (found.has(key)) {
      throw new PairingError(`Ambiguous pairing in ${folder}: '${path.basename(found.get(key))}' and '${name}' differ only by case`);
    }
    found.set(key, full);
  }
  return found;
}

/** Pair two folders by case-insensitive file name; returns ``{pairs, skipped}``. */
export function pairImages(referenceDir, candidateDir, extensions, skipUnmatched = false) {
  const refs = discoverImages(referenceDir, extensions);
  const cands = discoverImages(candidateDir, extensions);
  if (!refs.size) throw new PairingError(`No supported images (${extensions.join(', ')}) found in reference folder ${referenceDir}`);
  if (!cands.size) throw new PairingError(`No supported images (${extensions.join(', ')}) found in candidate folder ${candidateDir}`);

  const sortKeys = (keys) => [...keys].sort();
  const common = sortKeys([...refs.keys()].filter((k) => cands.has(k)));
  const onlyRef = sortKeys([...refs.keys()].filter((k) => !cands.has(k)));
  const onlyCand = sortKeys([...cands.keys()].filter((k) => !refs.has(k)));
  const skipped = {
    reference_without_candidate: onlyRef.map((k) => path.basename(refs.get(k))),
    candidate_without_reference: onlyCand.map((k) => path.basename(cands.get(k))),
  };
  if (onlyRef.length || onlyCand.length) {
    const lines = [];
    if (onlyRef.length) lines.push('  reference images with no candidate: ' + skipped.reference_without_candidate.join(', '));
    if (onlyCand.length) lines.push('  candidate images with no reference: ' + skipped.candidate_without_reference.join(', '));
    const message = 'Unmatched images (pairing is by case-insensitive file name):\n' + lines.join('\n');
    if (!skipUnmatched) throw new PairingError(message + '\nSet input.skip_unmatched: true (or pass --skip-unmatched) to ignore them.');
    logger.warning(`${message}\nThese files will be skipped.`);
  }
  if (!common.length) throw new PairingError('No matching file names between reference and candidate folders');
  const pairs = common.map((k) => ({ name: path.basename(refs.get(k)), reference: refs.get(k), candidate: cands.get(k) }));
  logger.info(`Found ${pairs.length} image pair(s)`);
  return { pairs, skipped };
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
export class BenchmarkRunner {
  constructor(config) {
    this.config = config;
  }

  /** Compute every metric for two already-preprocessed images. */
  scorePreprocessed(name, ref, cand) {
    const result = new PairResult(name, String(ref.meta.path ?? ''), String(cand.meta.path ?? ''));
    result.mask_source_reference = ref.maskReliable ? ref.maskSource : 'none';
    result.mask_source_candidate = cand.maskReliable ? cand.maskSource : 'none';
    result.refImage = ref;
    result.candImage = cand;
    const m = this.config.metrics;

    result.ssim = computeSsim(ref.rgb, cand.rgb, m.ssim);
    result.ssim_score = ssimToScore(result.ssim);

    result.silhouette_iou = computeSilhouetteIou(ref.hasMask ? ref.mask : null, cand.hasMask ? cand.mask : null);
    result.silhouette_score = iouToScore(result.silhouette_iou);
    if (result.silhouette_score === null) logger.info(`${name}: silhouette IoU not available (no reliable foreground mask on both images)`);

    const edge = computeEdgeSimilarity(ref.rgb, cand.rgb, m.edge);
    result.edge_score = edge.score;
    result.edge_chamfer_ref_to_cand = edge.chamferRefToCand;
    result.edge_chamfer_cand_to_ref = edge.chamferCandToRef;
    if (result.edge_score === null) logger.info(`${name}: edge score not available (no edges detected in either image)`);

    const scores = { ssim: result.ssim_score, silhouette: result.silhouette_score, edge: result.edge_score };
    const floors = this.config.score_floors;
    const gamma = this.config.score_gamma;
    result.calibrated_scores = Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, applyScoreFloor(v, floors[k] ?? 0, gamma)]));
    const { pairScore, effectiveWeights } = combineScores(result.calibrated_scores, this.config.weights);
    result.pair_score = pairScore;
    result.effective_weights = effectiveWeights;
    result.unavailable_metrics = METRIC_NAMES.filter((k) => scores[k] === null);
    return result;
  }

  /** Preprocess and score one pair; errors are captured in ``result.error``. */
  comparePair(reference, candidate, name = null, runDir = null) {
    name = name ?? path.basename(reference);
    const pcfg = this.config.preprocessing;
    const ocfg = this.config.output;
    let ref;
    let cand;
    try {
      ref = preprocessImage(reference, pcfg);
      cand = preprocessImage(candidate, pcfg);
    } catch (exc) {
      if (!(exc instanceof ImageLoadError)) throw exc;
      logger.error(`${name}: ${exc.message}`);
      const r = new PairResult(name, String(reference), String(candidate));
      r.error = exc.message;
      return r;
    }

    const aligned = alignCandidate(ref, cand, pcfg);
    cand = aligned.image;
    const info = aligned.info;
    const result = this.scorePreprocessed(name, ref, cand);
    result.alignment = info;
    if (info.applied && (info.shift_px[0] !== 0 || info.shift_px[1] !== 0)) {
      const sign = (v) => (v >= 0 ? '+' : '') + v;
      logger.info(
        `${name}: candidate aligned by (${sign(info.shift_px[0])}, ${sign(info.shift_px[1])}) px via ${info.method} ` +
          `(${(100 * info.shift_fraction[0]).toFixed(1)}%, ${(100 * info.shift_fraction[1]).toFixed(1)}% of canvas)`,
      );
    }

    if (runDir !== null) {
      if (ocfg.save_preprocessed) {
        const pRef = savePreprocessed(ref, path.join(runDir, 'preprocessed', 'reference'), name);
        const pCand = savePreprocessed(cand, path.join(runDir, 'preprocessed', 'candidate'), name);
        result.preprocessed_reference = path.relative(runDir, pRef);
        result.preprocessed_candidate = path.relative(runDir, pCand);
      }
      if (ocfg.save_comparisons) {
        const comp = reporting.saveComparisonImage(result, path.join(runDir, 'comparisons'));
        result.comparison_image = path.relative(runDir, comp);
      }
    }

    logger.info(
      `${name}: SSIM=${fmt(result.ssim, 4)} (${fmt(result.ssim_score, 1)})  ` +
        `IoU=${fmt(result.silhouette_iou, 4)}  Edge=${fmt(result.edge_score, 1)}  -> pair_score=${fmt(result.pair_score)}` +
        (result.unavailable_metrics.length ? `  [unavailable: ${result.unavailable_metrics.join(', ')}]` : ''),
    );
    return result;
  }

  /** Create ``outputRoot/run_YYYYMMDD_HHMMSS[_label]`` (suffixed ``_2``, ``_3`` ... if it exists). */
  static createRunDir(outputRoot, label = null) {
    const d = new Date();
    const p = (v) => String(v).padStart(2, '0');
    const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
    const base = `run_${stamp}` + (label ? `_${slugify(label, 90)}` : '');
    let runDir = path.join(outputRoot, base);
    let counter = 1;
    while (fs.existsSync(runDir)) {
      counter += 1;
      runDir = path.join(outputRoot, `${base}_${counter}`);
    }
    fs.mkdirSync(runDir, { recursive: true });
    return runDir;
  }

  /**
   * Run the benchmark over two folders. ``outputRoot`` null disables file
   * output unless ``runDir`` is given. ``meshComplexity`` / ``rig`` turn the
   * image score into ``shape_score`` and apply those penalty terms.
   */
  run(referenceDir, candidateDir, outputRoot, { skipUnmatched = null, runDir = null, meshComplexity = null, rig = null } = {}) {
    skipUnmatched = skipUnmatched ?? this.config.input.skip_unmatched;
    const { pairs, skipped } = pairImages(referenceDir, candidateDir, this.config.input.extensions, skipUnmatched);

    if (runDir !== null) fs.mkdirSync(runDir, { recursive: true });
    else if (outputRoot !== null) runDir = BenchmarkRunner.createRunDir(outputRoot);
    if (runDir !== null) logger.info(`Run directory: ${runDir}`);

    const results = pairs.map((p) => this.comparePair(p.reference, p.candidate, p.name, runDir));
    const power = this.config.view_power;
    const groups = computeGroupScores(results, this.config.output.group_separator, power);
    let overall;
    if (Object.keys(groups).length && power !== 1.0) {
      // views are combined per object first; objects are then averaged
      overall = mean(Object.values(groups).map((g) => g.score));
    } else {
      overall = computeOverallScore(results, power);
    }
    let shapeScore = null;
    if (meshComplexity !== null || rig !== null) {
      shapeScore = overall;
      overall = applyPenalties(overall, [meshComplexity, rig]);
      if (meshComplexity !== null) {
        const m = meshComplexity;
        logger.info(
          `Mesh faces: reference ${m.reference.faces}, candidate ${m.candidate.faces} ` +
            `(ratio ${m.face_ratio === null ? 'inf' : m.face_ratio.toPrecision(3)}) -> mesh score ${m.score.toFixed(1)}, weight ${m.weight.toFixed(2)}`,
        );
      }
      if (rig !== null) {
        if (rig.applicable) {
          const comps = Object.entries(rig.components).map(([k, v]) => `${k} ${v.toFixed(0)}`).join(', ');
          logger.info(`Rig score ${rig.score.toFixed(1)} (${comps}), weight ${rig.weight.toFixed(2)}`);
        } else {
          logger.info('Reference has no rig; rig term not applied');
        }
      }
    }
    const bench = new BenchmarkResult({
      pairs: results,
      overallScore: overall,
      runDir,
      config: this.config,
      skippedUnmatched: skipped,
      groupScores: groups,
      shapeScore,
      mesh: meshComplexity,
      rig,
    });
    for (const [group, info] of Object.entries(groups)) logger.info(`Group ${group}: score ${info.score.toFixed(2)} over ${info.num_pairs} pair(s)`);
    const failed = results.filter((r) => !r.ok).map((r) => r.name);
    if (failed.length) logger.warning(`${failed.length} pair(s) failed and were excluded from the overall score: ${failed.join(', ')}`);
    if (overall === null) logger.error('No valid pairs; overall score is not available');
    else logger.info(`Overall score: ${overall.toFixed(2)} over ${bench.validPairs.length} valid pair(s)`);

    if (runDir !== null) {
      reporting.saveMetricsJson(bench, path.join(runDir, 'metrics.json'));
      reporting.saveMetricsCsv(bench, path.join(runDir, 'metrics.csv'));
      reporting.saveReport(bench, runDir);
    }
    return bench;
  }
}

/** Lower-case, keep letters/digits/CJK/``._+``, join the rest with ``-``; used for folder names. */
export function slugify(text, maxLen = 40) {
  let t = String(text).trim().toLowerCase();
  t = t.replace(/[^0-9a-z一-鿿.+_]+/g, '-').replace(/^[-._]+|[-._]+$/g, '');
  t = t.replace(/-{2,}/g, '-');
  return t.slice(0, maxLen).replace(/[-._]+$/, '') || 'run';
}

/** Power mean of non-negative scores; 1 = arithmetic mean, smaller leans towards the minimum. */
export function powerMean(values, power = 1.0) {
  const arr = values.map((v) => Math.max(v, 0));
  if (power === 1.0) return mean(arr);
  return mean(arr.map((v) => v ** power)) ** (1.0 / power);
}

export function computeOverallScore(results, power = 1.0) {
  const valid = results.filter((r) => r.ok).map((r) => r.pair_score);
  return valid.length ? powerMean(valid, power) : null;
}

/** ``mug_front.png`` -> ``mug``; ``null`` when grouping is disabled or the stem has no separator. */
export function groupName(fileName, separator) {
  if (!separator) return null;
  const stem = path.parse(fileName).name;
  const idx = stem.indexOf(separator);
  if (idx < 0) return null;
  return stem.slice(0, idx) || null;
}

/** Power mean of valid pair scores per object prefix, sorted by group name. */
export function computeGroupScores(results, separator, power = 1.0) {
  const buckets = new Map();
  for (const r of results) {
    const g = groupName(r.name, separator);
    if (g === null || !r.ok) continue;
    if (!buckets.has(g)) buckets.set(g, []);
    buckets.get(g).push(r);
  }
  const out = {};
  for (const g of [...buckets.keys()].sort()) {
    const rs = buckets.get(g);
    out[g] = { score: powerMean(rs.map((r) => r.pair_score), power), num_pairs: rs.length, pairs: rs.map((r) => r.name) };
  }
  return out;
}
