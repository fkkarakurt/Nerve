/* worker.js - runs the discovery search off the main thread.
 *
 * The search is a tight blocking loop over hundreds of generations. On the
 * main thread that is a frozen tab; here it is just a worker doing its job,
 * posting a message every few generations while the page stays interactive.
 *
 * All the postMessage calls with type "progress" / "front" / "done" / "error"
 * are emitted from inside the WebAssembly module itself (see the EM_JS blocks
 * in discover_web.c), so this file only has to start things and get out of
 * the way.
 */
/* global importScripts, createNerveDiscover, postMessage, onmessage */

importScripts("nerve_discover.js");

let mod = null;

createNerveDiscover().then(function (m) {
  mod = m;
  postMessage({ type: "ready" });
}).catch(function (err) {
  postMessage({ type: "error", message: "could not start the engine: " + err });
});

onmessage = function (e) {
  const msg = e.data || {};
  if (msg.cmd !== "run") return;

  if (!mod) {
    postMessage({ type: "error", message: "the engine is still loading" });
    return;
  }

  let rows;
  try {
    rows = mod.ccall("ndw_load_csv", "number", ["string"], [msg.csv]);
  } catch (err) {
    postMessage({ type: "error", message: "could not read that data: " + err });
    return;
  }

  if (!rows) {
    postMessage({
      type: "error",
      message: mod.ccall("ndw_error", "string", [], [])
    });
    return;
  }

  const vars = mod.ccall("ndw_vars", "number", [], []);
  const names = [];
  for (let i = 0; i < vars; i++) {
    names.push(mod.ccall("ndw_var_name", "string", ["number"], [i]));
  }
  postMessage({ type: "loaded", rows: rows, vars: vars, names: names });

  try {
    mod.ccall(
      "ndw_run", null,
      ["number", "number", "number", "number", "number"],
      [msg.ops, msg.generations, msg.maxNodes, msg.population, msg.seed]
    );
  } catch (err) {
    postMessage({ type: "error", message: "the search stopped: " + err });
  }
};
