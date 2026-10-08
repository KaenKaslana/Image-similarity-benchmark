import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { alignCandidate, phaseCorrelate } from '../src/alignment.js';
import { BenchmarkRunner } from '../src/benchmark.js';
import { defaultConfig, validateConfig } from '../src/config.js';
import { Image8, loadImage, resizePil, savePng } from '../src/image.js';
import { preprocessDecoded } from '../src/preprocessing.js';
import { allOrientations, defaultRenderOptions, loadMesh, renderView } from '../src/render.js';
import { autoOrient, applyOrientation } from '../src/orient.js';
import { analyseRig, rigComparison } from '../src/rig.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BIN = path.join(HERE, '..', 'bin', 'imgsim.js');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'imgsim-test-'));

/** RGBA image with a filled disc; transparent elsewhere. */
function disc(size, cx, cy, r, value = 120) {
  const img = new Image8(size, size, 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const inside = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r;
      const o = (y * size + x) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = inside ? value : 0;
      img.data[o + 3] = inside ? 255 : 0;
    }
  }
  return img;
}

function writeSet(dir, images) {
  fs.mkdirSync(dir, { recursive: true });
  for (const [name, img] of Object.entries(images)) savePng(img, path.join(dir, name));
  return dir;
}

/** A unit cube as a binary STL. */
function writeCubeStl(file, scale = 1) {
  const v = (x, y, z) => [x * scale, y * scale, z * scale];
  const quads = [
    [v(0, 0, 0), v(0, 1, 0), v(1, 1, 0), v(1, 0, 0)],
    [v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)],
    [v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0)],
    [v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)],
    [v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)],
    [v(0, 1, 0), v(0, 1, 1), v(1, 1, 1), v(1, 1, 0)],
  ];
  const tris = quads.flatMap(([a, b, c, d]) => [[a, b, c], [a, c, d]]);
  const buf = Buffer.alloc(84 + tris.length * 50);
  buf.writeUInt32LE(tris.length, 80);
  tris.forEach((t, i) => {
    const o = 84 + i * 50 + 12;
    t.flat().forEach((f, k) => buf.writeFloatLE(f, o + k * 4));
  });
  fs.writeFileSync(file, buf);
  return file;
}

test('PNG round trip and Pillow-style resize', () => {
  const img = disc(40, 20, 20, 12);
  const file = path.join(tmp, 'disc.png');
  savePng(img, file);
  const back = loadImage(file);
  assert.equal(back.mode, 'RGBA');
  assert.ok(back.alpha !== null);
  assert.equal(back.alpha[20 * 40 + 20], 255);
  assert.equal(back.alpha[0], 0);
  const small = resizePil(new Image8(40, 40, 3, back.rgb), 20, 20, 'lanczos');
  assert.equal(small.width, 20);
  assert.equal(small.data[(10 * 20 + 10) * 3], 120);
});

test('preprocessing: alpha mask, canvas fit and foreground_bbox crop', () => {
  const cfg = validateConfig(defaultConfig());
  const dec = loadImage(path.join(tmp, 'disc.png'));
  const pre = preprocessDecoded(dec, cfg.preprocessing, 'disc.png');
  assert.equal(pre.rgb.width, 512);
  assert.equal(pre.maskSource, 'alpha');
  assert.ok(pre.hasMask);
  // transparent pixels composite onto the white background
  assert.equal(pre.rgb.data[0], 255);
  cfg.preprocessing.crop_mode = 'foreground_bbox';
  const cropped = preprocessDecoded(dec, cfg.preprocessing, 'disc.png');
  assert.ok(cropped.meta.foreground_fraction > pre.meta.foreground_fraction);
  assert.deepEqual(cropped.meta.crop_box.length, 4);
});

test('phase correlation recovers a known translation and alignment applies it', () => {
  const cfg = validateConfig(defaultConfig());
  const ref = preprocessDecoded(loadImage(path.join(tmp, 'disc.png')), cfg.preprocessing, 'a');
  savePng(disc(40, 26, 23, 12), path.join(tmp, 'disc2.png'));
  const cand = preprocessDecoded(loadImage(path.join(tmp, 'disc2.png')), cfg.preprocessing, 'b');
  const w = ref.rgb.width;
  const a = Float32Array.from(ref.mask);
  const b = Float32Array.from(cand.mask);
  const { shift } = phaseCorrelate(a, b, w, w);
  // 6 px right, 3 px down in a 40 px image -> *12.8 on the 512 canvas
  assert.ok(Math.abs(shift[0] - 6 * 12.8) < 1.5, `dx ${shift[0]}`);
  assert.ok(Math.abs(shift[1] - 3 * 12.8) < 1.5, `dy ${shift[1]}`);
  const { image, info } = alignCandidate(ref, cand, cfg.preprocessing);
  assert.equal(info.applied, true);
  assert.equal(info.signal, 'mask');
  assert.ok(info.shift_px[0] < -70 && info.shift_px[1] < -30);
  assert.ok(image !== cand);
  cfg.preprocessing.alignment = 'none';
  assert.equal(alignCandidate(ref, cand, cfg.preprocessing).info.applied, false);
});

test('BenchmarkRunner scores identical images 100 and writes the run outputs', () => {
  const cfg = validateConfig(defaultConfig());
  const refDir = writeSet(path.join(tmp, 'ref'), { 'mug_front.png': disc(64, 32, 32, 20), 'mug_top.png': disc(64, 32, 32, 24) });
  const candDir = writeSet(path.join(tmp, 'cand'), { 'mug_front.png': disc(64, 32, 32, 20), 'mug_top.png': disc(64, 30, 34, 18) });
  const result = new BenchmarkRunner(cfg).run(refDir, candDir, path.join(tmp, 'out'));
  assert.equal(result.pairs.length, 2);
  assert.ok(Math.abs(result.pairs[0].pair_score - 100) < 1e-9);
  assert.ok(result.pairs[1].pair_score < 100);
  assert.equal(result.group_scores.mug.num_pairs, 2);
  for (const f of ['metrics.json', 'metrics.csv', 'report.html', 'comparisons/mug_front.png', 'preprocessed/reference/mug_front_mask.png']) {
    assert.ok(fs.existsSync(path.join(result.run_dir, f)), f);
  }
  const json = JSON.parse(fs.readFileSync(path.join(result.run_dir, 'metrics.json'), 'utf-8'));
  assert.equal(json.num_valid_pairs, 2);
  assert.ok(!('lpips_score' in json.pairs['mug_front.png']));
});

test('unmatched files abort unless skipped', () => {
  const cfg = validateConfig(defaultConfig());
  const refDir = writeSet(path.join(tmp, 'ref2'), { 'a.png': disc(32, 16, 16, 8), 'b.png': disc(32, 16, 16, 8) });
  const candDir = writeSet(path.join(tmp, 'cand2'), { 'A.PNG': disc(32, 16, 16, 8) });
  assert.throws(() => new BenchmarkRunner(cfg).run(refDir, candDir, null), /Unmatched images/);
  const result = new BenchmarkRunner(cfg).run(refDir, candDir, null, { skipUnmatched: true });
  assert.equal(result.pairs.length, 1);
  assert.deepEqual(result.skipped_unmatched.reference_without_candidate, ['b.png']);
});

test('mesh loading, rendering and auto-orient on an STL cube', () => {
  const file = writeCubeStl(path.join(tmp, 'cube.stl'));
  const mesh = loadMesh(file);
  assert.equal(mesh.numFaces, 12);
  assert.equal(mesh.numVertices, 8);
  const opts = { ...defaultRenderOptions(), size: 64, supersample: 1 };
  const img = renderView(mesh, 'front', opts);
  // the cube fills 85 % of the canvas: centre opaque, corner transparent
  assert.equal(img.data[(32 * 64 + 32) * 4 + 3], 255);
  assert.equal(img.data[3], 0);
  const iso = renderView(mesh, 'iso', opts);
  assert.ok(iso.data[(32 * 64 + 32) * 4 + 3] > 0);
  assert.equal(allOrientations().length, 24);
  const best = autoOrient(mesh, mesh, ['front', 'side', 'top'], { size: 32, yawStep: 0 });
  assert.ok(best.mean_iou > 0.99);
  const turned = applyOrientation(mesh, '+y', '+z', 45);
  assert.equal(turned.numFaces, 12);
});

test('rig analysis of a model without a skeleton and the rig term', () => {
  const info = analyseRig(path.join(tmp, 'cube.stl'));
  assert.equal(info.has_skin, false);
  assert.equal(info.readable, true);
  const cfg = validateConfig(defaultConfig()).rig;
  const cmp = rigComparison(info, info, cfg);
  assert.equal(cmp.applicable, false);
  assert.equal(cmp.score, null);
});

test('CLI: compare on a folder and compare-models on two STL files', () => {
  const refDir = path.join(tmp, 'ref');
  const candDir = path.join(tmp, 'cand');
  const r = spawnSync(process.execPath, [BIN, 'compare', '--reference', refDir, '--candidate', candDir, '--no-save', '--log-level', 'ERROR'], { encoding: 'utf-8' });
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /mug_front\.png\s+1\.000\s+1\.000\s+100\.0\s+100\.00/);
  assert.match(r.stdout, /overall_score/);
  const big = writeCubeStl(path.join(tmp, 'cube2.stl'), 3);
  const m = spawnSync(
    process.execPath,
    [BIN, 'compare-models', '--reference', path.join(tmp, 'cube.stl'), '--candidate', big, '--output', path.join(tmp, 'out3d'), '--size', '64', '--views', 'front,top', '--auto-orient', '--log-level', 'ERROR'],
    { encoding: 'utf-8' },
  );
  assert.equal(m.status, 0, m.stderr);
  // scale does not matter: both cubes render identically
  assert.match(m.stdout, /overall_score\s+100\.00/);
  const runDir = fs.readdirSync(path.join(tmp, 'out3d'))[0];
  const models = JSON.parse(fs.readFileSync(path.join(tmp, 'out3d', runDir, 'models.json'), 'utf-8'));
  assert.equal(models.candidate.auto_orient.ranking.length > 0, true);
  assert.ok(fs.existsSync(path.join(tmp, 'out3d', runDir, 'renders', 'candidate', 'front.png')));
  const bad = spawnSync(process.execPath, [BIN, 'compare', '--reference', refDir], { encoding: 'utf-8' });
  assert.equal(bad.status, 2);
  assert.match(bad.stderr, /missing required option --candidate/);
});
