# Publishing a New Research Note

The public Ledger site is generated from source files in `_posts/`.

## 1. Create a Source Post

Create `_posts/YYYY-MM-DD-a-short-slug.md` for Markdown or `.html` for an HTML-based article. Include this front matter:

```yaml
---
ledger_published: true
title: "Your post title"
description: "A one-sentence summary used in the archive."
date: YYYY-MM-DD
topic: "Reinforcement learning"
image: "folder/figure.png" # optional; path relative to assets/img/
caption: "What the archive figure shows." # optional
---
```

`ledger_published: true` is required. Template and unfinished posts stay out of the public site by omitting it or setting it to `false`.

## 2. Build the Ledger Pages Locally

Run:

```powershell
node scripts/build-ledger-site.mjs
```

This reads every published post, regenerates `blog/`, updates `assets/ledger/posts.js`, and updates the article route map. With the preview server running, open `http://localhost:4173/research/` to inspect the archive and new article.

## 3. Publish

Commit and push the post source file. GitHub Actions runs the same generator before building and deploying the GitHub Pages site, so the production archive updates automatically.

For a post that needs complex interactive components, write it as `.html`; the generator preserves the article body. Standard Markdown posts support headings, paragraphs, lists, links, code blocks, figures, and MathJax equations.
