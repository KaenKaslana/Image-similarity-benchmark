/**
 * Output writers: metrics.json, metrics.csv, comparison images, report.html
 * and the console summary (port of src/reporting.py).
 *
 * The Python version draws its reports with matplotlib; here the comparison
 * images are plain PNG strips (reference | candidate | |difference|) and the
 * report is a self-contained HTML page next to them.
 */

import fs from 'node:fs';
import path from 'node:path';
import { Image8, savePng } from './image.js';
import { groupName } from './benchmark.js';
import { fmt, getLogger } from './util.js';

const logger = getLogger('reporting');

export const CSV_COLUMNS = [
  'name',
  'ssim',
  'ssim_score',
  'silhouette_iou',
  'silhouette_score',
  'edge_score',
  'pair_score',
  'unavailable_metrics',
  'mask_source_reference',
  'mask_source_candidate',
  'alignment_shift_px',
  'error',
  'reference_path',
  'candidate_path',
];

// ---------------------------------------------------------------------------
// Tabular outputs
// ---------------------------------------------------------------------------
export function saveMetricsJson(result, file) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(result.toDict(), null, 2), 'utf-8');
  logger.info(`Wrote ${file}`);
  return file;
}

/** Python ``str(float)``-style number formatting for the CSV. */
function csvValue(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return Number.isInteger(v) ? `${v}.0` : String(v);
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function csvRow(row) {
  return CSV_COLUMNS.map((c) => csvValue(row[c])).join(',');
}

/** One row per pair, then ``__group__``, ``__shape__``, ``__rig__`` / ``__mesh__`` and ``__overall__`` rows. */
export function saveMetricsCsv(result, file) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const lines = [CSV_COLUMNS.join(',')];
  for (const p of result.pairs) {
    const row = {};
    for (const c of CSV_COLUMNS) row[c] = p[c] ?? null;
    row.unavailable_metrics = p.unavailable_metrics.join(';');
    row.alignment_shift_px = p.alignment && p.alignment.applied ? `${p.alignment.shift_px[0]};${p.alignment.shift_px[1]}` : '';
    lines.push(csvRow(row));
  }
  for (const [group, info] of Object.entries(result.group_scores)) lines.push(csvRow({ name: `__group__:${group}`, pair_score: info.score }));
  if (result.mesh !== null || result.rig !== null) lines.push(csvRow({ name: '__shape__', pair_score: result.shape_score }));
  if (result.rig !== null) {
    const r = result.rig;
    lines.push(
      csvRow({
        name: '__rig__',
        pair_score: r.score,
        unavailable_metrics: r.applicable ? '' : 'reference has no rig',
        reference_path: rigBrief(r.reference),
        candidate_path: rigBrief(r.candidate),
      }),
    );
    lines.push(
      csvRow({
        name: '__mesh__',
        pair_score: result.mesh.score,
        reference_path: `faces=${result.mesh.reference.faces}`,
        candidate_path: `faces=${result.mesh.candidate.faces}`,
      }),
    );
  }
  lines.push(csvRow({ name: '__overall__', pair_score: result.overall_score }));
  fs.writeFileSync(file, lines.join('\r\n') + '\r\n', 'utf-8');
  logger.info(`Wrote ${file}`);
  return file;
}

/** ``bones=58 clips=1 motion=0.067`` for the CSV and captions. */
export function rigBrief(info) {
  if (!info) return '';
  if (info.readable === false) return 'unreadable';
  if (!info.has_skin) return 'no rig';
  let s = `bones=${info.joints} clips=${(info.clips ?? []).length}`;
  if (info.motion_amplitude !== null && info.motion_amplitude !== undefined) s += ` motion=${info.motion_amplitude.toFixed(3)}`;
  if (info.deformation_ok === false) s += ' BROKEN';
  return s;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------
// matplotlib "magma" sampled at 17 points, linearly interpolated.
const MAGMA = [
  [0, 0, 4], [10, 8, 34], [29, 17, 71], [54, 16, 107], [81, 18, 124], [106, 28, 129], [131, 38, 129], [156, 46, 127],
  [183, 55, 121], [208, 65, 111], [231, 82, 99], [245, 107, 92], [252, 137, 97], [254, 167, 114], [254, 196, 136],
  [253, 226, 163], [252, 253, 191],
];

function magma(v) {
  const x = Math.min(1, Math.max(0, v)) * (MAGMA.length - 1);
  const i = Math.min(MAGMA.length - 2, Math.floor(x));
  const t = x - i;
  return MAGMA[i].map((c, k) => Math.round(c + (MAGMA[i + 1][k] - c) * t));
}

/** Strip of reference | candidate | |difference| (magma), separated by white gaps. */
export function comparisonStrip(ref, cand) {
  const w = ref.width;
  const h = ref.height;
  const gap = 8;
  const out = new Image8(3 * w + 2 * gap, h, 3);
  out.data.fill(255);
  const W = out.width;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const s = (y * w + x) * 3;
      let diff = 0;
      for (let c = 0; c < 3; c++) {
        out.data[(y * W + x) * 3 + c] = ref.data[s + c];
        out.data[(y * W + x + w + gap) * 3 + c] = cand.data[s + c];
        diff += Math.abs(ref.data[s + c] - cand.data[s + c]);
      }
      const col = magma(diff / 3 / 255);
      for (let c = 0; c < 3; c++) out.data[(y * W + x + 2 * (w + gap)) * 3 + c] = col[c];
    }
  }
  return out;
}

export function saveComparisonImage(p, directory) {
  fs.mkdirSync(directory, { recursive: true });
  const file = path.join(directory, `${path.parse(p.name).name}.png`);
  savePng(comparisonStrip(p.refImage.rgb, p.candImage.rgb), file);
  return file;
}

// ---------------------------------------------------------------------------
// HTML report
// ---------------------------------------------------------------------------
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** ``mug_front.png`` -> ``front`` (or the whole stem when there is no separator). */
export function viewName(fileName, separator) {
  const stem = path.parse(fileName).name;
  if (separator && stem.includes(separator)) return stem.slice(stem.indexOf(separator) + separator.length) || stem;
  return stem;
}

function metricsText(p) {
  const lines = [
    `SSIM: ${fmt(p.ssim, 4)}  (score ${fmt(p.ssim_score)})`,
    `Silhouette IoU: ${fmt(p.silhouette_iou, 4)}  (score ${fmt(p.silhouette_score)})`,
    `Edge score: ${fmt(p.edge_score)}`,
    `Pair score: ${fmt(p.pair_score)}`,
  ];
  const raw = { ssim: p.ssim_score, silhouette: p.silhouette_score, edge: p.edge_score };
  const cal = p.calibrated_scores;
  if (Object.entries(raw).some(([k, v]) => cal[k] !== v)) {
    lines.splice(3, 0, `after floors: ssim ${fmt(cal.ssim, 0)} sil ${fmt(cal.silhouette, 0)} edge ${fmt(cal.edge, 0)}`);
  }
  if (p.unavailable_metrics.length) lines.push('not available: ' + p.unavailable_metrics.join(', '));
  if (p.alignment && p.alignment.method !== 'none') {
    const [dx, dy] = p.alignment.shift_px;
    const sign = (v) => (v >= 0 ? '+' : '') + v;
    lines.push(`aligned: shift (${sign(dx)}, ${sign(dy)}) px, ${p.alignment.applied ? 'applied' : 'NOT applied'}`);
  }
  if (p.error) lines.push(`ERROR: ${p.error}`);
  return lines.join('\n');
}

function meshCaption(result) {
  const lines = [];
  if (result.mesh === null && result.rig === null) return lines;
  let text = `shape score ${fmt(result.shape_score)}`;
  if (result.mesh !== null) {
    const m = result.mesh;
    text +=
      ` | faces: reference ${m.reference.faces.toLocaleString('en-US')}, candidate ${m.candidate.faces.toLocaleString('en-US')}` +
      ` (x${fmt(m.face_ratio, 3)}) -> mesh score ${fmt(m.score)} at weight ${m.weight.toFixed(2)}`;
    if (m.bonus_weight) text += `, efficiency bonus +${(100 * m.bonus_weight * m.bonus).toFixed(1)} %`;
  }
  lines.push(text);
  if (result.rig !== null) {
    const r = result.rig;
    lines.push(
      r.applicable
        ? `rig score ${fmt(r.score)} at weight ${r.weight.toFixed(2)} | reference ${rigBrief(r.reference)} | candidate ${rigBrief(r.candidate)}`
        : `rig: reference has no rig (not applied); candidate ${rigBrief(r.candidate)}`,
    );
  }
  return lines;
}

/** Red -> yellow -> green background for a 0-100 score (grey for n/a). */
function scoreColor(score) {
  if (score === null || score === undefined) return '#ebebeb';
  const t = Math.min(100, Math.max(0, score)) / 100;
  const hue = 120 * t;
  return `hsl(${hue.toFixed(0)}, 60%, 82%)`;
}

/**
 * Write ``report.html``: a score table (one row per object, one column per
 * view when file names are grouped) and one row per pair with its
 * comparison strip and metrics.
 */
export function saveReport(result, runDir) {
  const pairs = result.pairs;
  if (!pairs.length) return [];
  const separator = String(result.config.output?.group_separator ?? '_');
  const parts = [];
  const title = `Image similarity benchmark - overall score: ${fmt(result.overall_score)}`;
  parts.push(`<h1>${esc(title)}</h1>`);
  parts.push(`<p class="sub">${result.validPairs.length}/${pairs.length} valid pairs`);
  for (const line of meshCaption(result)) parts.push(`<br>${esc(line)}`);
  parts.push('</p>');

  if (Object.keys(result.group_scores).length) {
    const views = [];
    const table = {};
    for (const p of pairs) {
      const g = groupName(p.name, separator);
      if (g === null) continue;
      const v = viewName(p.name, separator);
      if (!views.includes(v)) views.push(v);
      (table[g] ??= {})[v] = p.ok ? p.pair_score : null;
    }
    parts.push('<table class="scores"><thead><tr><th>object</th>');
    for (const v of views) parts.push(`<th>${esc(v)}</th>`);
    parts.push('<th>object score</th></tr></thead><tbody>');
    for (const [g, info] of Object.entries(result.group_scores)) {
      parts.push(`<tr><th>${esc(g)}</th>`);
      for (const v of views) {
        const s = table[g]?.[v] ?? null;
        parts.push(`<td style="background:${scoreColor(s)}">${fmt(s)}</td>`);
      }
      parts.push(`<td class="obj" style="background:${scoreColor(info.score)}">${fmt(info.score)}</td></tr>`);
    }
    parts.push(`<tr><th>overall</th>${views.map(() => '<td></td>').join('')}`);
    parts.push(`<td class="obj" style="background:${scoreColor(result.overall_score)}">${fmt(result.overall_score)}</td></tr>`);
    parts.push('</tbody></table>');
  }

  parts.push('<div class="pairs">');
  for (const p of pairs) {
    parts.push('<section class="pair">');
    parts.push(`<h2>${esc(p.name)} <span style="background:${scoreColor(p.pair_score)}">${fmt(p.pair_score)}</span></h2>`);
    if (p.comparison_image) {
      const src = p.comparison_image.split(path.sep).map(encodeURIComponent).join('/');
      parts.push(`<figure><img src="${src}" alt="${esc(p.name)}"><figcaption>reference | candidate | |difference|</figcaption></figure>`);
    }
    parts.push(`<pre>${esc(metricsText(p))}</pre></section>`);
  }
  parts.push('</div>');

  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark report</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;color:#1d1d1f;background:#fff}
h1{font-size:20px;margin:0 0 4px} .sub{color:#555;margin:0 0 16px;font-size:14px;line-height:1.5}
table.scores{border-collapse:collapse;margin-bottom:24px;font-variant-numeric:tabular-nums}
table.scores th,table.scores td{border:1px solid #ccc;padding:6px 12px;text-align:center}
td.obj{font-weight:600}
.pair{border-top:1px solid #ddd;padding:12px 0}
.pair h2{font-size:16px;margin:0 0 8px} .pair h2 span{padding:2px 8px;border-radius:4px;font-variant-numeric:tabular-nums}
figure{margin:0 0 8px} img{max-width:100%;height:auto;border:1px solid #eee}
figcaption{font-size:12px;color:#666}
pre{font-size:13px;margin:0;white-space:pre-wrap}
</style></head><body>
${parts.join('\n')}
</body></html>
`;
  const file = path.join(runDir, 'report.html');
  fs.writeFileSync(file, html, 'utf-8');
  logger.info(`Wrote report: ${file}`);
  return [file];
}

// ---------------------------------------------------------------------------
// Console summary
// ---------------------------------------------------------------------------
const L = (s, n) => String(s).slice(0, n).padEnd(n);
const R = (s, n) => String(s).padStart(n);

export function formatSummaryTable(result) {
  const header = `${L('name', 24)} ${R('SSIM', 7)} ${R('IoU', 7)} ${R('Edge', 7)} ${R('Pair', 7)}`;
  const rule = '-'.repeat(header.length);
  const lines = [header, rule];
  for (const p of result.pairs) {
    if (p.error) {
      lines.push(`${L(p.name, 24)} ERROR: ${p.error}`);
      continue;
    }
    lines.push(`${L(p.name, 24)} ${R(fmt(p.ssim, 3), 7)} ${R(fmt(p.silhouette_iou, 3), 7)} ${R(fmt(p.edge_score, 1), 7)} ${R(fmt(p.pair_score, 2), 7)}`);
  }
  lines.push(rule);
  const blank = `${R('', 7)} ${R('', 7)} ${R('', 7)}`;
  for (const [group, info] of Object.entries(result.group_scores)) {
    lines.push(`${L(`[${group}] (${info.num_pairs} views)`, 24)} ${blank} ${R(fmt(info.score), 7)}`);
  }
  if (Object.keys(result.group_scores).length) lines.push(rule);
  if (result.mesh !== null) {
    const m = result.mesh;
    lines.push(`${L('shape_score', 24)} ${blank} ${R(fmt(result.shape_score), 7)}`);
    const faces = `faces ${m.reference.faces.toLocaleString('en-US')} vs ${m.candidate.faces.toLocaleString('en-US')} (x${fmt(m.face_ratio, 3)})`;
    lines.push(`${L('mesh_score', 24)} ${faces.padEnd(23)} ${R(fmt(m.score), 7)}   weight ${m.weight.toFixed(2)}`);
    if (m.bonus_weight) {
      lines.push(`${L('mesh_bonus', 24)} fewer faces: +${(100 * m.bonus_weight * m.bonus).toFixed(1)} % of ${(100 * m.bonus_weight).toFixed(0)} %`);
    }
  }
  if (result.rig !== null) {
    const r = result.rig;
    if (result.mesh === null) lines.push(`${L('shape_score', 24)} ${blank} ${R(fmt(result.shape_score), 7)}`);
    if (r.applicable) {
      const comps = Object.entries(r.components).map(([k, v]) => `${k.slice(0, 4)} ${v.toFixed(0)}`).join(' ');
      lines.push(`${L('rig_score', 24)} ${comps.padEnd(23)} ${R(fmt(r.score), 7)}   weight ${r.weight.toFixed(2)}`);
      lines.push(`${L('  reference', 24)} ${rigBrief(r.reference)}`);
      lines.push(`${L('  candidate', 24)} ${rigBrief(r.candidate)}`);
    } else {
      lines.push(`${L('rig_score', 24)} reference has no rig (not applied); candidate: ${rigBrief(r.candidate)}`);
    }
  }
  if (result.mesh !== null || result.rig !== null) lines.push(rule);
  lines.push(`${L('overall_score', 24)} ${blank} ${R(fmt(result.overall_score), 7)}`);
  return lines.join('\n');
}
