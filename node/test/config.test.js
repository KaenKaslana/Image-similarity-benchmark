import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { ConfigError, configFromDict, loadConfig, validateWeights } from '../src/config.js';
import { blendWithShapeScore, complexityScore, efficiencyBonus, meshComplexity, applyPenalties } from '../src/complexity.js';
import { computeGroupScores, groupName, powerMean, slugify } from '../src/benchmark.js';
import { optimalDftSize, fft2d } from '../src/fft.js';
import { roundHalfEven } from '../src/util.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

test('the repository configs load and drop the LPIPS weight', () => {
  for (const name of ['default.yaml', 'shape.yaml']) {
    const file = path.join(ROOT, 'configs', name);
    if (!fs.existsSync(file)) continue;
    const cfg = loadConfig(file);
    const total = Object.values(cfg.weights).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(total - 1) < 1e-9, `${name}: weights sum to ${total}`);
    assert.ok(!('lpips' in cfg.weights));
    assert.ok(!('lpips' in cfg.score_floors));
  }
  const shape = loadConfig(path.join(ROOT, 'configs', 'shape.yaml'));
  // 0.45 / 0.25 / 0.00 renormalised over 0.70
  assert.ok(Math.abs(shape.weights.silhouette - 0.45 / 0.7) < 1e-12);
  assert.equal(shape.weights.ssim, 0);
  assert.equal(shape.preprocessing.crop_mode, 'foreground_bbox');
  assert.equal(shape.mesh_complexity.mode, 'fewer_is_better');
});

test('unknown keys, bad enums and bad weights are rejected', () => {
  assert.throws(() => configFromDict({ preprocessing: { canvas: 3 } }), ConfigError);
  assert.throws(() => configFromDict({ preprocessing: { crop_mode: 'tight' } }), ConfigError);
  assert.throws(() => configFromDict({ weights: { ssim: 0.5, silhouette: 0.5, edge: 0.5 } }), ConfigError);
  assert.throws(() => configFromDict({ weights: { ssim: 1 } }), ConfigError);
  assert.throws(() => configFromDict({ weights: { lpips: 1, ssim: 0, silhouette: 0, edge: 0 } }), ConfigError);
  assert.throws(() => configFromDict({ score_floors: { lpips: 70, blur: 1 } }), ConfigError);
  assert.throws(() => configFromDict({ mesh_complexity: { free_log2: 3, zero_log2: 2 } }), ConfigError);
  assert.deepEqual(validateWeights({ ssim: 0.2, silhouette: 0.5, edge: 0.3 }), { ssim: 0.2, silhouette: 0.5, edge: 0.3 });
});

test('complexity score and penalties', () => {
  assert.equal(complexityScore(1, 1, 5), 100);
  assert.equal(complexityScore(2, 1, 5), 100);
  assert.equal(complexityScore(32, 1, 5), 0);
  assert.ok(Math.abs(complexityScore(8, 1, 5) - 50) < 1e-9);
  assert.ok(Math.abs(complexityScore(1 / 8, 1, 5) - 50) < 1e-9);
  assert.equal(complexityScore(1 / 8, 1, 5, 'fewer_is_better'), 100);
  assert.equal(efficiencyBonus(0.25, 2), 1);
  assert.equal(efficiencyBonus(2, 2), 0);
  const cfg = { weight: 0.15, free_log2: 1, zero_log2: 5, mode: 'fewer_is_better', bonus_weight: 0.05, bonus_log2: 2 };
  const same = meshComplexity({ faces: 1000, vertices: 600 }, { faces: 1000, vertices: 500 }, cfg);
  assert.equal(same.score, 100);
  assert.equal(blendWithShapeScore(80, same), 80);
  const bloated = meshComplexity({ faces: 1000, vertices: 600 }, { faces: 32000, vertices: 500 }, cfg);
  assert.ok(Math.abs(blendWithShapeScore(80, bloated) - 80 * 0.85) < 1e-9);
  const lean = meshComplexity({ faces: 1000, vertices: 600 }, { faces: 250, vertices: 500 }, cfg);
  assert.ok(Math.abs(blendWithShapeScore(80, lean) - 80 * 1.05) < 1e-9);
  assert.equal(blendWithShapeScore(99, lean), 100);
  assert.equal(applyPenalties(80, [null, { applicable: false, score: 0, weight: 1 }, bloated]), blendWithShapeScore(80, bloated));
});

test('grouping, power mean and slugify', () => {
  assert.equal(groupName('mug_front.png', '_'), 'mug');
  assert.equal(groupName('front.png', '_'), null);
  assert.equal(groupName('front.png', ''), null);
  const pairs = [
    { name: 'mug_front.png', pair_score: 100, ok: true },
    { name: 'mug_top.png', pair_score: 0, ok: true },
    { name: 'chair_front.png', pair_score: 50, ok: true },
    { name: 'chair_top.png', pair_score: 50, error: 'x', ok: false },
  ];
  const groups = computeGroupScores(pairs, '_', 1.0);
  assert.deepEqual(Object.keys(groups), ['chair', 'mug']);
  assert.equal(groups.mug.score, 50);
  assert.equal(groups.chair.num_pairs, 1);
  assert.ok(powerMean([100, 0], 0.25) < 50);
  assert.equal(powerMean([40, 60], 1), 50);
  assert.equal(slugify('Victorian Chair (v2)'), 'victorian-chair-v2');
  assert.equal(slugify('   '), 'run');
  assert.equal(slugify('a'.repeat(50), 10), 'aaaaaaaaaa');
});

test('FFT round trip and optimal DFT sizes', () => {
  assert.equal(optimalDftSize(500), 500);
  assert.equal(optimalDftSize(513), 540);
  assert.equal(optimalDftSize(512), 512);
  const rows = 6;
  const cols = 10; // 2 * 3 and 2 * 5 cover the non-power-of-two radices
  const re = Float64Array.from({ length: rows * cols }, (_, i) => Math.sin(i) + (i % 7));
  const im = new Float64Array(rows * cols);
  const orig = re.slice();
  fft2d(re, im, rows, cols);
  // DC term equals the sum of the input
  assert.ok(Math.abs(re[0] - orig.reduce((a, b) => a + b, 0)) < 1e-9);
  fft2d(re, im, rows, cols, true);
  for (let i = 0; i < re.length; i++) {
    assert.ok(Math.abs(re[i] / (rows * cols) - orig[i]) < 1e-9);
    assert.ok(Math.abs(im[i]) < 1e-9);
  }
});

test('roundHalfEven matches numpy rint', () => {
  assert.equal(roundHalfEven(0.5), 0);
  assert.equal(roundHalfEven(1.5), 2);
  assert.equal(roundHalfEven(2.5), 2);
  assert.equal(roundHalfEven(-0.5), -0);
  assert.equal(roundHalfEven(-1.5), -2);
  assert.equal(roundHalfEven(2.4999), 2);
  assert.equal(roundHalfEven(-2.6), -3);
});
