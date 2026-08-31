/*
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Nerve Discover — give it numbers, get back an equation.
//
// Unlike the rest of this package there is no model here and nothing to
// download: equation discovery is pure algorithm, so the module is a few tens
// of kilobytes and starts instantly. Everything runs in-process.
import createDiscover from "./discover.mjs";

/** Operator flags, mirroring nerve_discover.h. */
export const Op = Object.freeze({
  ADD: 1 << 0, SUB: 1 << 1, MUL: 1 << 2, DIV: 1 << 3,
  NEG: 1 << 4, INV: 1 << 5, SQR: 1 << 6, SQRT: 1 << 7,
  EXP: 1 << 8, LOG: 1 << 9, SIN: 1 << 10, COS: 1 << 11, TANH: 1 << 12,
});

/** Ready-made operator sets. */
export const Ops = Object.freeze({
  ARITH:   Op.ADD | Op.SUB | Op.MUL | Op.DIV,
  ALGEBRA: Op.ADD | Op.SUB | Op.MUL | Op.DIV | Op.INV | Op.SQR | Op.SQRT,
  DEFAULT: Op.ADD | Op.SUB | Op.MUL | Op.DIV | Op.INV | Op.SQR | Op.SQRT |
           Op.SIN | Op.COS | Op.EXP | Op.LOG,
  ALL:     Op.ADD | Op.SUB | Op.MUL | Op.DIV | Op.INV | Op.SQR | Op.SQRT |
           Op.SIN | Op.COS | Op.EXP | Op.LOG | Op.NEG | Op.TANH,
});

let _modulePromise = null;

/** Load the engine once; later calls reuse it. */
export async function load(moduleOptions = {}) {
  if (!_modulePromise) _modulePromise = createDiscover(moduleOptions);
  return new Discoverer(await _modulePromise);
}

/**
 * Parse a CSV-ish table: inputs in the leading columns, the value to explain
 * in the last. A first row that is not all numbers is taken as the header.
 * Accepts comma, semicolon or tab separators.
 */
export function parseTable(text) {
  const rows = [];
  let names = null;

  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const cells = line.split(/[,;\t]/).map((c) => c.trim());
    if (cells.length < 2) continue;

    const nums = cells.map(Number);
    if (nums.some((v) => !Number.isFinite(v))) {
      if (names === null && rows.length === 0) names = cells.slice(0, -1);
      continue;                       // header, or an unusable row
    }
    rows.push(nums);
  }

  if (rows.length === 0) throw new Error("no numeric rows found");
  const width = rows[0].length;
  const clean = rows.filter((r) => r.length === width);
  const nvars = width - 1;
  if (nvars < 1) throw new Error("need at least one input column");

  const X = new Float64Array(clean.length * nvars);
  const y = new Float64Array(clean.length);
  clean.forEach((r, i) => {
    for (let j = 0; j < nvars; j++) X[i * nvars + j] = r[j];
    y[i] = r[nvars];
  });
  if (!names || names.length !== nvars) {
    names = Array.from({ length: nvars }, (_, j) => "x" + j);
  }
  return { X, y, names, rows: clean.length, nvars };
}

export class Discoverer {
  constructor(M) { this._M = M; }

  /** The engine's version string. */
  get version() {
    return this._M.ccall("nd_json_version", "string", [], []);
  }

  /**
   * Discover an equation.
   *
   * @param {object} data  { X, y, names } — X row-major, or pass a CSV string
   *                       to `fromTable` first.
   * @param {object} opts  { ops, generations, maxNodes, population, seed,
   *                         kneeSlack }
   * @returns {{ knee: Equation, front: Equation[] }}
   */
  fit(data, opts = {}) {
    const M = this._M;
    const { X, y, names } = data;
    const nvars = names ? names.length : (X.length / y.length) | 0;
    const n = y.length;

    if (!(n >= 4)) throw new Error("need at least 4 rows");
    if (X.length < n * nvars) throw new Error("X is smaller than n * nvars");

    const {
      ops = Ops.DEFAULT, generations = 250, maxNodes = 24,
      population = 300, seed = 1, kneeSlack = 0.02,
    } = opts;

    const xs = X instanceof Float64Array ? X : Float64Array.from(X);
    const ys = y instanceof Float64Array ? y : Float64Array.from(y);

    const px = M._malloc(xs.length * 8);
    const py = M._malloc(ys.length * 8);
    let pn = 0;
    try {
      M.HEAPF64.set(xs, px >> 3);
      M.HEAPF64.set(ys, py >> 3);
      if (names && names.length) {
        const joined = names.join("\n");
        const bytes = M.lengthBytesUTF8(joined) + 1;
        pn = M._malloc(bytes);
        M.stringToUTF8(joined, pn, bytes);
      }

      const json = M.ccall(
        "nd_json_fit", "string",
        ["number", "number", "number", "number", "number", "number",
         "number", "number", "number", "number", "number"],
        [px, py, n, nvars, pn, ops, generations, maxNodes, population,
         seed, kneeSlack]
      );
      const out = JSON.parse(json);
      if (out.error) throw new Error(out.error);
      return out;
    } finally {
      M._free(px); M._free(py); if (pn) M._free(pn);
    }
  }

  /** Convenience: discover straight from a CSV string. */
  fromTable(text, opts = {}) {
    return this.fit(parseTable(text), opts);
  }
}

export default { load, parseTable, Op, Ops, Discoverer };
