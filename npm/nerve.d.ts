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

// Type definitions for Nerve — on-device AI in WebAssembly.

export interface GenerateOptions {
  /** number of tokens to generate (default 128) */
  steps?: number;
  /** sampling temperature; 0 = greedy (default 0.85) */
  temperature?: number;
  /** nucleus (top-p) threshold (default 0.9) */
  topP?: number;
  /** RNG seed for reproducible output */
  seed?: number;
  /** called with each decoded text fragment as it is produced */
  onToken?: (token: string) => void;
}

export interface Example { text: string; label: string; }

export interface Classification {
  label: string;
  confidence: number;
  scores: Record<string, number>;
}

export interface SearchHit { text: string; score: number; }

/**
 * Nerve — a real Transformer text generator + a MiniLM sentence-embedding
 * encoder, running entirely in WebAssembly (browser or Node). No server, no GPU.
 */
export declare class Nerve {
  /** embedding dimensionality of the encoder */
  readonly dim: number;

  /** Load the models and return a ready instance. */
  static load(moduleOptions?: Record<string, unknown>): Promise<Nerve>;

  /** Sentence -> L2-normalised meaning vector. */
  embed(text: string): Float32Array;

  /** Cosine similarity (-1..1) between two sentences. */
  similarity(a: string, b: string): number;

  /** Generate text from a prompt (streams via opts.onToken). */
  generate(prompt: string, opts?: GenerateOptions): string;

  /** Train a tiny classifier on your own labelled examples, on-device. */
  teach(examples: Example[], opts?: { epochs?: number; lr?: number }): Nerve;

  /** Classify text using the classifier from teach(). */
  classify(text: string): Classification;

  /** Build a semantic index over notes for search(). */
  index(notes: string[]): Nerve;

  /** Search indexed notes by meaning; returns top-K hits. */
  search(query: string, topK?: number): SearchHit[];
}

export default Nerve;
