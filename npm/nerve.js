// Nerve — run a real LLM + sentence embeddings in the browser or Node.
// Zero-dependency neural nets in pure C, compiled to WebAssembly.
import createNerve from "./nerve.mjs";

export class Nerve {
  constructor(M) {
    this._M = M;
    this.dim = M.ccall("nerve_web_dim", "number", [], []);
    this._head = null;
    this._index = null;
  }

  /** Load the models (decoder + MiniLM encoder). Returns a ready Nerve. */
  static async load(moduleOptions = {}) {
    const M = await createNerve(moduleOptions);
    const rc = M.ccall("nerve_web_init", "number", [], []);
    if (rc !== 0) throw new Error("Nerve init failed (code " + rc + ")");
    return new Nerve(M);
  }

  /** Sentence -> a meaning vector (Float32Array, L2-normalised). */
  embed(text) {
    const M = this._M, d = this.dim;
    const p = M.ccall("nerve_web_embed", "number", ["string"], [text]);
    return Float32Array.from(M.HEAPF32.subarray(p >> 2, (p >> 2) + d));
  }

  /** Cosine similarity (-1..1) between two sentences. */
  similarity(a, b) {
    const x = this.embed(a), y = this.embed(b);
    let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i];
    return s;
  }

  /** Generate text from a prompt. Streams via opts.onToken if given. */
  generate(prompt, opts = {}) {
    const { steps = 128, temperature = 0.85, topP = 0.9,
            seed = (Math.random() * 1e9) | 0, onToken = null } = opts;
    const M = this._M;
    M.ccall("nerve_web_gen_start", null,
      ["string", "number", "number", "number", "number"],
      [prompt, steps, temperature, topP, seed]);
    let out = "";
    while (!M.ccall("nerve_web_gen_done", "number", [], [])) {
      const t = M.ccall("nerve_web_gen_step", "string", [], []);
      out += t; if (onToken) onToken(t);
    }
    return out;
  }

  /** Train a tiny classifier on your own labelled examples (on-device). */
  teach(examples, opts = {}) {
    const { epochs = 300, lr = 0.3 } = opts, d = this.dim;
    const labels = [...new Set(examples.map(e => e.label))], NC = labels.length;
    const X = examples.map(e => this.embed(e.text)), y = examples.map(e => labels.indexOf(e.label));
    const W = new Float32Array(d * NC), b = new Float32Array(NC);
    for (let e = 0; e < epochs; e++) for (let s = 0; s < X.length; s++) {
      const lo = new Float32Array(NC);
      for (let c = 0; c < NC; c++) { let v = b[c]; for (let i = 0; i < d; i++) v += W[i*NC+c]*X[s][i]; lo[c] = v; }
      let mx = -1e9; for (let c = 0; c < NC; c++) if (lo[c] > mx) mx = lo[c];
      let sum = 0; const pr = new Float32Array(NC);
      for (let c = 0; c < NC; c++) { pr[c] = Math.exp(lo[c]-mx); sum += pr[c]; }
      for (let c = 0; c < NC; c++) pr[c] /= sum;
      for (let c = 0; c < NC; c++) { const g = pr[c]-(c===y[s]?1:0); b[c]-=lr*g; for (let i = 0; i < d; i++) W[i*NC+c]-=lr*g*X[s][i]; }
    }
    this._head = { W, b, NC, labels };
    return this;
  }

  /** Classify text using the head from teach(). */
  classify(text) {
    if (!this._head) throw new Error("call teach() first");
    const h = this._head, x = this.embed(text), d = this.dim, lo = new Float32Array(h.NC);
    for (let c = 0; c < h.NC; c++) { let v = h.b[c]; for (let i = 0; i < d; i++) v += h.W[i*h.NC+c]*x[i]; lo[c] = v; }
    let mx = -1e9; for (let c = 0; c < h.NC; c++) if (lo[c] > mx) mx = lo[c];
    let sum = 0; const pr = new Float32Array(h.NC);
    for (let c = 0; c < h.NC; c++) { pr[c] = Math.exp(lo[c]-mx); sum += pr[c]; }
    for (let c = 0; c < h.NC; c++) pr[c] /= sum;
    let bi = 0; for (let c = 1; c < h.NC; c++) if (pr[c] > pr[bi]) bi = c;
    const scores = {}; h.labels.forEach((l, i) => scores[l] = pr[i]);
    return { label: h.labels[bi], confidence: pr[bi], scores };
  }

  /** Build a semantic index over notes for search(). */
  index(notes) {
    this._index = notes.map(t => ({ text: t, vec: this.embed(t) }));
    return this;
  }

  /** Search the indexed notes by meaning; returns top-K hits. */
  search(query, topK = 3) {
    if (!this._index) throw new Error("call index() first");
    const q = this.embed(query);
    return this._index
      .map(n => { let s = 0; for (let i = 0; i < q.length; i++) s += q[i]*n.vec[i]; return { text: n.text, score: s }; })
      .sort((a, b) => b.score - a.score).slice(0, topK);
  }
}

export default Nerve;
