// The static explorer's server: answers the page's /api fetches in the browser from the files in data/ (written by
// explorer/export_static.py), computing lens, connections, hover, web, web summary and search the way server.py and
// runtime.py do. Routes that need the model, the agent or a write don't exist here and answer 404. Loaded before
// app.js; every other fetch goes out unchanged.
const DATA = new URL("./data/", import.meta.url);
const realFetch = window.fetch.bind(window);
const files = new Map();
// each data file is fetched once; a failed fetch is retried with backoff and never cached, so a dropped connection
// costs a retry, not the rest of the session
async function fetchData(path, kind) {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await realFetch(new URL(path, DATA));
      if (!r.ok) throw new Error(`${path} ${r.status}`);
      return kind === "json" ? await r.json() : await r.arrayBuffer();
    } catch (err) {
      if (attempt >= 3) throw err;
      await new Promise(res => setTimeout(res, 250 * 2 ** attempt));
    }
  }
}
const load = (path, kind = "json") => {
  if (!files.has(path)) files.set(path, fetchData(path, kind).catch(err => { files.delete(path); throw err; }));
  return files.get(path);
};

// ---------- data ----------
const shape = () => load("shape.json");
const index = () => load("index.json");
const vocab = () => load("vocab.json");
const words = () => load("words.json");  // {layer: {neuron: top 12 words}} for the swept neurons
const hasRun = async l => String(l) in (await index()).layers;

// float16 -> float32, exact, through a table
const HALF = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  HALF[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

// lens.bin: 3-byte token ids [N, K] for steer up, steer down, logit up, logit down; then KL up, down float32 [N]
const lensCache = new Map();
function lensLayer(l) {
  if (!lensCache.has(l)) lensCache.set(l, Promise.all([shape(), load(`L${l}/lens.bin`, "bin")]).then(([sh, buf]) => {
    const N = sh.N, K = sh.k_lens, b = new Uint8Array(buf), per = N * K;
    const ids = j => { const a = new Uint32Array(per), o = j * per * 3; for (let i = 0; i < per; i++) a[i] = b[o + 3 * i] | (b[o + 3 * i + 1] << 8) | (b[o + 3 * i + 2] << 16); return a; };
    const f = new Float32Array(buf, 4 * per * 3, 2 * N);
    return { K, steer: { up: ids(0), down: ids(1) }, logit: { up: ids(2), down: ids(3) }, kl_up: f.subarray(0, N), kl_down: f.subarray(N) };
  }).catch(err => { lensCache.delete(l); throw err; }));
  return lensCache.get(l);
}
const toks = (v, arr, K, n, k) => Array.from(arr.subarray(n * K, n * K + k), t => v[t]);

// con.bin (one layer, k links) and con2.bin (every layer, 2 links): per field, down then up
function parseCon(buf, off, N, k) {
  const m = N * k, f = {};
  const take = (Type, bytes) => { const a = new Type(buf, off, m); off += m * bytes; return a; };
  f.z = { down: take(Uint16Array, 2), up: take(Uint16Array, 2) };
  f.n = { down: take(Int16Array, 2), up: take(Int16Array, 2) };
  f.layer = { down: take(Int8Array, 1), up: take(Int8Array, 1) };
  f.sign = { down: take(Int8Array, 1), up: take(Int8Array, 1) };
  return [{ k, ...f }, off];
}
const conCache = new Map();
const conLayer = l => {
  if (!conCache.has(l)) conCache.set(l, Promise.all([shape(), load(`L${l}/con.bin`, "bin")]).then(([sh, buf]) => parseCon(buf, 0, sh.N, sh.k_con)[0]).catch(err => { conCache.delete(l); throw err; }));
  return conCache.get(l);
};
let con2P = null;
const con2 = () => con2P ||= Promise.all([shape(), load("con2.bin", "bin")]).then(([sh, buf]) => {
  const out = []; let off = 0;
  for (let l = 0; l < sh.L; l++) { const [c, o] = parseCon(buf, off, sh.N, 2); out.push(c); off = o; }
  return out;
}).catch(err => { con2P = null; throw err; });

// Python's round(): exact halves go to the even neighbour (float16 link scores often land on them)
const pyRound = x => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r; };
const r2 = x => pyRound(x * 100) / 100, r3 = x => pyRound(x * 1000) / 1000;
const pyStrip = s => s.replace(/^\s+|\s+$/g, "");

// ---------- runtime.lens / runtime.connections ----------
async function lens(layer, n, k) {
  const [v, L] = await Promise.all([vocab(), lensLayer(layer)]);
  return { up: toks(v, L.logit.up, L.K, n, k), down: toks(v, L.logit.down, L.K, n, k),
           steer_up: toks(v, L.steer.up, L.K, n, k), steer_down: toks(v, L.steer.down, L.K, n, k),
           kl_up: r3(L.kl_up[n]), kl_down: r3(L.kl_down[n]) };
}
async function connections(layer, n, k, lensK) {
  const c = await conLayer(layer), out = {};
  for (const side of ["down", "up"]) {
    out[side] = [];
    for (let i = 0; i < k; i++) {
      const j = n * c.k + i, z = HALF[c.z[side][j]];
      if (z > -1e4) out[side].push({ layer: c.layer[side][j], n: c.n[side][j], z: r2(z), sign: c.sign[side][j] });
    }
  }
  if (lensK) {
    const all = [...out.down, ...out.up], v = await vocab();
    const Ls = new Map(await Promise.all([...new Set(all.map(x => x.layer))].map(async l => [l, await lensLayer(l)])));
    for (const x of all) {
      const L = Ls.get(x.layer);
      x.lens = toks(v, L.steer.up, L.K, x.n, lensK); x.lens_neg = toks(v, L.steer.down, L.K, x.n, lensK);
      x.logit = toks(v, L.logit.up, L.K, x.n, lensK); x.logit_neg = toks(v, L.logit.down, L.K, x.n, lensK);
    }
  }
  return out;
}

// ---------- server.cone_links / web / web summary ----------
async function coneLinks(layer, n, k = 2, zmin = 15, cap = 20000, first = null, hops = null) {
  const [own, c2] = await Promise.all([conLayer(layer), con2()]);
  const out = [];
  for (const [side, step] of [["down", 1], ["up", -1]]) {
    let front = [[layer, n]], hop = 1;
    const seen = new Set([`${layer}/${n}`]), start = out.length;
    while (front.length && out.length - start < Math.floor(cap / 2) && (!hops || hop <= hops)) {
      const nxt = [];
      for (const [a, b] of front) {
        const c = hop === 1 ? own : c2[a], K = c.k, lim = hop === 1 ? (first || K) : k;
        for (let i = 0; i < lim && i < K; i++) {
          const j = b * K + i, z = HALF[c.z[side][j]];
          if (z < -1e4 || (hop > 1 && z < zmin)) break;
          const t = [c.layer[side][j], c.n[side][j]];
          const [src, dst] = side === "down" ? [[a, b], t] : [t, [a, b]];
          out.push([...src, ...dst, z * c.sign[side][j], hop * step]);
          const key = `${t[0]}/${t[1]}`;
          if (!seen.has(key)) { seen.add(key); nxt.push(t); }
        }
      }
      front = nxt; hop++;
    }
    out.length = Math.min(out.length, start + Math.floor(cap / 2));
  }
  return out;
}
async function web(q) {
  const links = await coneLinks(+q.layer, +q.n, +(q.k ?? 2), +(q.zmin ?? 15), +(q.cap ?? 20000), +(q.first ?? 5), +(q.hops ?? 0) || null);
  const rows = new Int16Array(links.length * 6);
  links.forEach((r, i) => rows.set([r[0], r[1], r[2], r[3], pyRound(r[4] * 10), r[5]], i * 6));
  return new Response(rows.buffer, { headers: { "Content-Type": "application/octet-stream", "X-Web-Total": String(links.length) } });
}
// each token's base rate in a layer: its share among all that layer's neurons' top-8 steer tokens (server._lens_base),
// counted when the layer is first needed
const dfCache = new Map();
function baseRate(l, L, t) {
  if (!dfCache.has(l)) {
    const df = new Map(), N = L.kl_up.length;
    for (const arr of [L.steer.up, L.steer.down])
      for (let m = 0; m < N; m++) for (let i = 0; i < 8; i++) { const x = arr[m * L.K + i]; df.set(x, (df.get(x) || 0) + 1); }
    dfCache.set(l, { df, N });
  }
  const b = dfCache.get(l);
  return ((b.df.get(t) || 0) + 1) / (2 * b.N);
}
async function webSummary(q) {
  const layer = +q.layer, n = +q.n, top = +(q.top ?? 30);
  const links = await coneLinks(layer, n, +(q.k ?? 2), 15, 20000, +(q.first ?? 5), +(q.hops ?? 0) || null);
  const v = await vocab();
  const sign = new Map([[`${layer}/${n}`, 1]]), reach = new Map();
  let zmax = 1.0;
  for (const r of links) zmax = Math.max(zmax, Math.abs(r[4]));
  for (const [sl, sn, dl, dn, z, hop] of links) {
    const [near, far] = hop > 0 ? [`${sl}/${sn}`, `${dl}/${dn}`] : [`${dl}/${dn}`, `${sl}/${sn}`];
    if (sign.has(far) || !sign.has(near)) continue;
    sign.set(far, sign.get(near) * (z > 0 ? 1 : -1));
    reach.set(far, [hop, Math.abs(z)]);
  }
  const Ls = new Map(await Promise.all([...new Set([...reach.keys()].map(k => +k.split("/")[0]))].map(async l => [l, await lensLayer(l)])));
  const pool = (nodes, flip) => {
    const acc = new Map();
    for (const [key, [hop, z]] of nodes) {
      const [l, m] = key.split("/").map(Number), L = Ls.get(l);
      const w = z / zmax * 0.8 ** (Math.abs(hop) - 1), arr = sign.get(key) * flip > 0 ? L.steer.up : L.steer.down;
      for (let i = 0; i < 8; i++) {
        const t = arr[m * L.K + i], s = v[t], kk = pyStrip(s).toLowerCase();
        if (!kk || kk.includes("�")) continue;
        if (!acc.has(kk)) acc.set(kk, { n: 0, w: 0, base: 0, forms: new Map() });
        const a = acc.get(kk), form = pyStrip(s);
        a.n++; a.w += w; a.base += baseRate(l, L, t); a.forms.set(form, (a.forms.get(form) || 0) + 1);
      }
    }
    for (const a of acc.values()) {
      const lift = (a.n / Math.max(1, nodes.length)) / (a.base / a.n);
      a.score = lift > 1 ? a.w * Math.log(lift) : 0;
    }
    return [...acc.values()].filter(a => a.score > 0).sort((x, y) => y.score - x.score).slice(0, top).map(a => {
      let best = null, bc = -1; for (const [f, c] of a.forms) if (c > bc) { best = f; bc = c; }
      return { tok: best, n: a.n, w: r2(a.score) };
    });
  };
  const all = [...reach.entries()], down = all.filter(([, x]) => x[0] > 0), up = all.filter(([, x]) => x[0] < 0);
  return { src: "steer", links: links.length,
           down: { neurons: down.length, toward: pool(down, 1), away: pool(down, -1) },
           up: { neurons: up.length, fed_by: pool(up, 1), held_by: pool(up, -1) } };
}

// ---------- server.search ----------
async function search(q0, limit) {
  const q = pyStrip(q0).toLowerCase();
  if (q.length < 2) return { ready: true, results: [] };
  const [v, sh, W, idx] = await Promise.all([vocab(), shape(), words(), index()]);
  const match = new Uint8Array(v.length);
  let nMatch = 0;
  v.forEach((t, i) => {
    const s = pyStrip(t);
    if (t.toLowerCase().includes(q) || (q.length >= 5 && s.length >= Math.max(4, Math.floor(q.length / 2)) && q.startsWith(s.toLowerCase()))) {
      if (!match[i]) { match[i] = 1; nMatch++; }
    }
  });
  const hits = [];
  for (const [l, ws] of Object.entries(W)) for (const [n, list] of Object.entries(ws))
    if (list.some(w => w.toLowerCase().includes(q))) hits.push([+l, +n, list]);
  if (!nMatch && !hits.length) return { ready: true, results: [], tokens: 0, total: 0 };
  let out = hits.map(([l, n, ws]) => [-1, -ws.length, l, n, "+", "words"]);
  if (nMatch) {
    const Ls = await Promise.all(Array.from({ length: sh.L }, (_, l) => lensLayer(l)));
    for (const [sign, which, src] of [["+", "up", "lens"], ["-", "down", "lens"], ["+", "up", "steer"], ["-", "down", "steer"]]) {
      Ls.forEach((L, l) => {
        const arr = (src === "lens" ? L.logit : L.steer)[which];
        for (let n = 0; n < sh.N; n++) {
          let count = 0, best = 99;
          for (let i = 0; i < L.K; i++) if (match[arr[n * L.K + i]]) { count++; if (i < best) best = i; }
          if (count) out.push([best, -count, l, n, sign, src]);
        }
      });
    }
  }
  const cmp = (a, b) => { for (let i = 0; i < 6; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1; return 0; };
  out.sort(cmp);
  const seen = new Set();
  out = out.filter(r => { const k = `${r[2]}/${r[3]}/${r[4]}`; if (seen.has(k)) return false; seen.add(k); return true; });
  const names = idx.names || {}, res = [];
  for (const [rank, , l, n, sign, src] of out.slice(0, limit)) {
    let lensToks;
    if (src === "words") lensToks = W[l][n].slice(0, 6);
    else { const L = await lensLayer(l); lensToks = toks(v, (src === "lens" ? L.logit : L.steer)[sign === "+" ? "up" : "down"], L.K, n, 6); }
    res.push({ layer: l, n, sign, rank: Math.max(rank, 0), src, name: names[`L${l}/N${n}`] || "", swept: !!(W[l] && String(n) in W[l]), lens: lensToks });
  }
  return { ready: true, results: res, tokens: nMatch, total: out.length };
}

// ---------- sweep routes ----------
async function unit(layer, n, kind) {
  const idx = await index(), out = { layer, n, kind, name: (idx.names || {})[`L${layer}/N${n}`] || "" };
  if (!(await hasRun(layer))) return out;
  const [stats, meta] = await Promise.all([load(`L${layer}/stats.json`), load(`L${layer}/meta.json`)]);
  const rec = stats.find(u => u.neuron === n && u.kind === kind) || null;
  const keep = r => ({ prompt: r.prompt, dose: r.dose, rel_push: r.rel_push, text: r.text, finish: r.finish, n_gen: r.n_gen });
  const rows = rec ? (await load(`L${layer}/rows.json`)).filter(r => r.kind === kind && r.neuron === n).map(keep) : [];
  const neurons = stats.filter(u => u.kind === "neuron");
  const rank = rec ? neurons.findIndex(u => u.neuron === n) + 1 : 0;
  return { ...out, run: meta.run, prompts: meta.prompts, rollouts: rows, baselines: meta.baselines, stats: rec,
           rank: rec ? (rank || null) : null, n_ranked: neurons.length };
}

const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
const ROUTES = {
  "/api/index": async () => { const { names, ...rest } = await index(); return rest; },
  "/api/status": () => ({ status: "ready" }),
  "/api/units": async q => (await hasRun(q.layer)) ? load(`L${+q.layer}/units.json`) : [],
  "/api/unit": q => unit(+q.layer, +q.n, q.kind || "neuron"),
  "/api/rollout": async q => { const r = (await load(`L${+q.layer}/rows.json`))[+q.i];
    return { kind: r.kind, neuron: r.neuron, prompt: r.prompt, dose: r.dose, rel_push: r.rel_push, text: r.text }; },
  "/api/embeddings": async q => (await hasRun(q.layer)) ? load(`L${+q.layer}/umap.json`).catch(() => ({})) : {},
  "/api/layout": q => (q.kind || "writes") === "writes" ? load("layout_writes.json") : { available: false },
  "/api/lens": q => lens(+q.layer, +q.n, 12),
  "/api/hover": async q => {
    const [l, c] = await Promise.all([lens(+q.layer, +q.n, 8), connections(+q.layer, +q.n, 16, 0)]);
    return { lens: l.up, lens_neg: l.down, steer: l.steer_up, steer_neg: l.steer_down, con: c };
  },
  "/api/connections": async q => {
    const out = await connections(+q.layer, +q.n, +(q.k ?? 24), +(q.lens ?? 5)), [W, idx] = await Promise.all([words(), index()]);
    for (const c of [...out.down, ...out.up]) { c.name = (idx.names || {})[`L${c.layer}/N${c.n}`] || ""; c.words = (W[c.layer]?.[c.n] || []).slice(0, 4); }
    return out;
  },
  "/api/search": q => search(q.q || "", +(q.limit ?? 300)),
  "/api/web/summary": q => webSummary(q),
  "/api/neuron/heads": () => ({ layers: {}, roles: {} }),  // needs the weights, as in the server's browse-only mode
  "/api/head": q => ({ layer: +q.layer, head: +q.head, full: false, src: "steer", reads: [], writes: [], q: [], k: [],
                       read_toks: [], write_toks: [], q_toks: [], k_toks: [] }),
};

window.fetch = async (input, init) => {
  const url = new URL(typeof input === "string" ? input : input.url, location.href);
  if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) return realFetch(input, init);
  const method = (init?.method || "GET").toUpperCase(), q = Object.fromEntries(url.searchParams);
  try {
    if (method === "GET" && url.pathname === "/api/web") return await web(q);
    const route = method === "GET" && ROUTES[url.pathname];
    if (!route) return json({ detail: "not available in the static explorer" }, 404);
    return json(await route(q));
  } catch (err) {
    return json({ detail: String(err.message || err) }, 500);
  }
};
