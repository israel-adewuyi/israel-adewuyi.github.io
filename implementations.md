# 2026-07-11

## Research Ledger Site

- The production site is a static Research Ledger implementation intended for GitHub Pages. The visible archives are generated, not rendered by the original al-folio post layout.
- `_posts/` remains the source of truth for articles. Only posts with `ledger_published: true` are included in the public Ledger archive.
- Run `node scripts/build-ledger-site.mjs` after changing a public post or Ledger presentation code. It regenerates `blog/`, the root static pages, `assets/ledger/posts.js`, and `assets/ledger/routes.js`.
- `scripts/build-ledger-site.mjs` is the central renderer. It handles Markdown/HTML post normalization, citations, footnotes, Markdown tables, heading IDs, nested contents numbering, and article output. Do not hand-edit generated pages under `blog/`.
- The public navigation has one Research archive. `/writing/` remains as a backward-compatible route that renders the same Research page, but Writing is not a visible navigation item.
- `assets/ledger/site-runtime.js` owns the home, Research, Publications, About, and Profile page content. Home uses focus themes with supporting opinions; About and Profile are intentionally minimal Coming soon pages.
- The generated pages include build-versioned Ledger asset URLs to avoid stale browser JavaScript/CSS during local preview and after deployment.

## Articles

- Article contents lists are generated automatically from `h2` through `h4`. Nested headings use parent-scoped numbering, such as `02`, `02.01`, and `02.01.01`.
- Citations and `d-footnote` tags are converted into numbered references and footnotes. Keep citations in the source posts; the generator formats them.
- The RLVR reward-landscape article uses self-contained interactive SVG charts generated from `assets/json/rlvr_landscape/correlations.json`. `assets/ledger/article-charts.js` adds hover and keyboard inspection; it deliberately avoids an external chart dependency.

## Deployment

- `.github/workflows/deploy.yml` installs Node 20 and runs `node scripts/build-ledger-site.mjs` before Jekyll builds and deploys GitHub Pages. The workflow must continue to include changes to `.mjs`, `scripts/`, `assets/`, post sources, and Ledger HTML/JS/CSS.
- Publish future posts by creating `_posts/YYYY-MM-DD-slug.md` or `.html`, adding `ledger_published: true` and the metadata described in `PUBLISHING.md`, then committing and pushing. GitHub Actions generates and deploys the public site.
- `_design_drafts/` and `server-preview.out.log` / `server-preview.err.log` are local draft/preview artifacts. Do not include them in the production commit unless deliberately preserving design history.
