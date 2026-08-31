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

// Type definitions for Nerve Discover — symbolic regression in WebAssembly.

/** Operator flags, mirroring nerve_discover.h. */
export declare const Op: Readonly<{
  ADD: number; SUB: number; MUL: number; DIV: number;
  NEG: number; INV: number; SQR: number; SQRT: number;
  EXP: number; LOG: number; SIN: number; COS: number; TANH: number;
}>;

/** Ready-made operator sets. Fewer operators, smaller haystack. */
export declare const Ops: Readonly<{
  /** `+ - * /` only. */
  ARITH: number;
  /** arithmetic plus `x^2`, `sqrt(x)`, `1/x` — the set for a design formula. */
  ALGEBRA: number;
  /** algebra plus `sin cos exp log`. */
  DEFAULT: number;
  /** everything, including `tanh` and unary minus. */
  ALL: number;
}>;

/** One point on the accuracy/complexity front. */
export interface Equation {
  /** number of nodes in the expression tree */
  nodes: number;
  /** coefficient of determination on the data it was fitted to */
  r2: number;
  /** root-mean-square error on the same data */
  rmse: number;
  /** the formula, e.g. `y = 0.996748*(m2*m1)/r^2 + 0.00943491` */
  equation: string;
}

export interface DiscoverResult {
  /**
   * The equation you probably want: the simplest one that gives up no more
   * than `kneeSlack` of the accuracy the front spans.
   */
  knee: Equation;
  /**
   * The whole front — the best equation found at every complexity, simplest
   * first — so you can see what each extra term bought.
   */
  front: Equation[];
}

export interface DiscoverOptions {
  /** which operators the search may use (default `Ops.DEFAULT`) */
  ops?: number;
  /** search iterations (default 250) */
  generations?: number;
  /** complexity ceiling (default 24) */
  maxNodes?: number;
  /** candidates per island (default 300) */
  population?: number;
  /** same seed, same answer (default 1) */
  seed?: number;
  /** fraction of the front's accuracy span the knee may concede (default 0.02) */
  kneeSlack?: number;
}

/** Data in the shape `fit` wants. */
export interface TableData {
  /** inputs, row-major: `X[i * nvars + j]` */
  X: Float64Array | number[];
  /** the value to explain, one per row */
  y: Float64Array | number[];
  /** column names, used in the printed equation */
  names?: string[];
  rows?: number;
  nvars?: number;
}

/**
 * Parse a CSV-ish table: inputs in the leading columns, the value to explain
 * in the last. A first row that is not all numbers is taken as the header.
 */
export declare function parseTable(text: string): Required<TableData>;

export declare class Discoverer {
  /** the engine's version string */
  readonly version: string;
  /** Discover an equation from arrays. */
  fit(data: TableData, opts?: DiscoverOptions): DiscoverResult;
  /** Discover an equation straight from a CSV string. */
  fromTable(text: string, opts?: DiscoverOptions): DiscoverResult;
}

/** Load the engine. There is no model to fetch, so this is fast. */
export declare function load(moduleOptions?: object): Promise<Discoverer>;

declare const _default: {
  load: typeof load;
  parseTable: typeof parseTable;
  Op: typeof Op;
  Ops: typeof Ops;
  Discoverer: typeof Discoverer;
};
export default _default;
