# LFM Neuron Explorer

A read-only, fully static explorer of the MLP neurons of [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B):
24 layers, 7,680 neurons each. It has the 3D map, layer views, search over every neuron's steer lens and logit lens,
the connection web, steered examples for the 9,216 neurons that were swept (384 per layer), embeddings of every
steered answer, and a few persona neurons to start from.

There is no server. Everything the page shows was precomputed into `data/`, and `static-api.js` answers the page's
requests in the browser. Nothing on the site runs a model, saves anything, or accepts input beyond browsing.

## Pages

- `index.html`: the landing page (landing.css, landing.js, hero.webp, og.jpg, favicon.svg)
- `explore.html`: the explorer itself; links like `explore.html#L12/N1473` open a neuron directly
- `404.html`: served by Render for unknown paths

## Preview locally

    python3 -m http.server 8000
    # open http://localhost:8000 (the 404 page only shows on Render)

## Deploy on Render (free static site)

1. Create a new GitHub repository and push this folder to it as the repository root.
2. In Render, choose **New → Static Site** and connect that repository.
3. Settings:
   - **Branch:** `main`
   - **Root directory:** leave empty
   - **Build command:** `echo "static site, nothing to build"` (or leave empty if Render allows it)
   - **Publish directory:** `.`
4. Create the site. Render serves it at `https://<name>.onrender.com`.

Instead of steps 2 and 3, you can use **New → Blueprint** with the included `render.yaml`.

Every later push to the branch redeploys the site.

## Rebuilding the data

The site is generated from the neuron-explorer repo's sweep results:

    cd neuron-explorer
    EXPLORER_MODEL=../models/LFM3-1.7B uv run python explorer/export_static.py --out ../lfm-neuron-explorer-site

This rebuilds `data/` from scratch and recopies the page files.

## What's in `data/`

| file | holds |
|---|---|
| `index.json`, `shape.json` | model, architecture, which layers have sweeps |
| `vocab.json` | token strings |
| `layout_writes.json` | each neuron's place on its layer's sheet |
| `words.json` | the distinctive words of every swept neuron's steered answers |
| `con2.bin` | every neuron's two strongest links each way, for the connection web |
| `L{n}/lens.bin` | steer lens and logit lens tokens, and steer KL, for every neuron of the layer |
| `L{n}/con.bin` | each neuron's 24 strongest direct-path links each way |
| `L{n}/units.json`, `stats.json`, `meta.json` | the layer's ranked swept neurons |
| `L{n}/rows.json` | every steered answer of the layer's sweep |
| `L{n}/umap.json` | the embeddings view |

Model outputs are shown exactly as the model wrote them, unfiltered.
