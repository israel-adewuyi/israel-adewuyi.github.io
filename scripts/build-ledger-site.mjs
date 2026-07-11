import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { resolve, basename, extname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const postsDirectory = resolve(root, "_posts");
const outputDirectory = resolve(root, "blog");
const ledgerDirectory = resolve(root, "assets", "ledger");
const assetVersion = Date.now().toString(36);

const existingPostDetails = {
  "rlvr-reward-landscape": { topic: "Reinforcement learning", image: "rlvr_landscape/rlvr_reward_landscape.png", caption: "A local two-dimensional tangent slice around a policy trained with GRPO." },
  "slim-peft": { topic: "Reinforcement learning / Finetuning", image: "rl_subnet_1/fft_fisher.png", caption: "Sparse parameter-selection experiments during RL finetuning." },
  "attn_sink_evidence": { topic: "Mechanistic interpretability", image: "resid_stream_with_attn_mlp.png", caption: "The residual stream: the common pathway through a transformer block." },
  "ner_with_rl": { topic: "Reinforcement learning", image: "graphrag/pipeline.png", caption: "The project treats structured extraction as a reward-design problem." },
  "replicating-graphrag": { topic: "Retrieval systems", image: "graphrag/pipeline.png", caption: "A GraphRAG pipeline from source documents to graph-based query answers." },
  "replicating-refusal": { topic: "Mechanistic interpretability", image: "refusal_replication/2-2b_resid_attrib_plot.png", caption: "Residual-stream attribution measured across layers in a Gemma 2 model." }
};

const pageRoutes = {
  home: "/", research: "/research/", writing: "/writing/", publications: "/publications/", about: "/about/", cv: "/cv/"
};
const correlationData = JSON.parse(await readFile(resolve(root, "assets", "json", "rlvr_landscape", "correlations.json"), "utf8"));

const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
const escapeAttribute = escapeHtml;
const cleanValue = (value = "") => value.trim().replace(/^['"]|['"]$/g, "").replace(/''/g, "'");
const citeTagPattern = /<d-cite\b([^>]*)>[\s\S]*?<\/d-cite\s*>/gi;
const footnoteTagPattern = /<d-footnote\b[^>]*>([\s\S]*?)<\/d-footnote\s*>/gi;

function parseFrontMatter(source) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) return { fields: {}, body: source };
  const fields = {};
  for (const line of match[1].split(/\r?\n/)) {
    const field = line.match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
    if (field) fields[field[1]] = cleanValue(field[2]);
  }
  return { fields, body: source.slice(match[0].length) };
}

function renderInline(value, cite, footnote) {
  const tokens = [];
  const protect = (html) => `@@LEDGER_TOKEN_${tokens.push(html) - 1}@@`;
  const protectedValue = value
    .replace(citeTagPattern, (_, attributes) => protect(cite(attributes.match(/\bkey="([^"]+)"/)?.[1] || "")))
    .replace(footnoteTagPattern, (_, note) => protect(footnote(renderInline(note, cite, footnote))))
    .replace(/<\/?a\b[^>]*>/gi, (tag) => protect(tag));
  return escapeHtml(protectedValue)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
    .replace(/@@LEDGER_TOKEN_(\d+)@@/g, (_, index) => tokens[Number(index)]);
}

function renderMarkdown(source, cite, footnote) {
  const prepared = source
    .replace(/<style[\s\S]*?<\/style\s*>/gi, "")
    .replace(/<script[\s\S]*?<\/script\s*>/gi, "")
    .replace(/\{\{\s*['"]([^'"]+)['"]\s*\|\s*relative_url\s*\}\}/g, "$1")
    .replace(/\{%\s*include\s+figure\.liquid([\s\S]*?)%\}/g, (_, attributes) => {
    const path = attributes.match(/path="([^"]+)"/)?.[1] || "";
    const alt = attributes.match(/alt="([^"]+)"/)?.[1] || "";
    return `<figure><img src="/${path.replace(/^\//, "")}" alt="${escapeAttribute(alt)}" /><figcaption>${escapeHtml(alt)}</figcaption></figure>`;
  });
  const lines = prepared.split(/\r?\n/);
  const output = [];
  let paragraph = [];
  let list = [];
  let code = [];
  let equation = [];
  let table = [];
  let rawBlock = [];
  let rawBlockTag = "";
  let inCode = false;
  let inEquation = false;
  const flushParagraph = () => { if (paragraph.length) { output.push(`<p>${renderInline(paragraph.join(" "), cite, footnote)}</p>`); paragraph = []; } };
  const flushList = () => { if (list.length) { output.push(`<ul>${list.map((item) => `<li>${renderInline(item, cite, footnote)}</li>`).join("")}</ul>`); list = []; } };
  const flushCode = () => { if (code.length) { output.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`); code = []; } };
  const flushEquation = () => { if (equation.length) { output.push(`<div class="equation">$$${escapeHtml(equation.join("\n"))}$$</div>`); equation = []; } };
  const tableCells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.map(tableCells);
    const separator = rows[1]?.every((cell) => /^:?-{3,}:?$/.test(cell));
    const headers = rows[0] || [];
    const body = rows.slice(separator ? 2 : 1);
    output.push(`<table><thead><tr>${headers.map((cell) => `<th>${renderInline(cell, cite, footnote)}</th>`).join("")}</tr></thead><tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell, cite, footnote)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
    table = [];
  };
  const flushRawBlock = () => { if (rawBlock.length) { output.push(normaliseHtml(rawBlock.join("\n"), cite, footnote)); rawBlock = []; rawBlockTag = ""; } };
  for (const line of lines) {
    if (rawBlockTag) {
      rawBlock.push(line);
      if (new RegExp(`</${rawBlockTag}\\s*>`, "i").test(line)) flushRawBlock();
      continue;
    }
    if (line.trim().startsWith("```")) { if (inCode) { flushCode(); inCode = false; } else { flushParagraph(); flushList(); flushTable(); inCode = true; } continue; }
    if (inCode) { code.push(line); continue; }
    if (line.trim() === "$$") { if (inEquation) { flushEquation(); inEquation = false; } else { flushParagraph(); flushList(); flushTable(); inEquation = true; } continue; }
    if (inEquation) { equation.push(line); continue; }
    if (!line.trim()) { flushParagraph(); flushList(); flushTable(); continue; }
    if (/^\s*\|.*\|\s*$/.test(line)) { flushParagraph(); flushList(); table.push(line); continue; }
    const heading = line.match(/^(#{2,4})\s+(.+)$/);
    if (heading) { flushParagraph(); flushList(); flushTable(); const level = heading[1].length; output.push(`<h${level}>${renderInline(heading[2], cite, footnote)}</h${level}>`); continue; }
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) { flushParagraph(); flushTable(); list.push(bullet[1]); continue; }
    const rawTag = line.trim().match(/^<(figure|table|div|details|blockquote|section)\b/i)?.[1]?.toLowerCase();
    if (rawTag) {
      flushParagraph(); flushList(); flushTable();
      if (new RegExp(`</${rawTag}\\s*>`, "i").test(line)) output.push(normaliseHtml(line, cite, footnote));
      else { rawBlock = [line]; rawBlockTag = rawTag; }
      continue;
    }
    if (/^<[^>]+>/.test(line.trim())) { flushParagraph(); flushList(); flushTable(); output.push(normaliseHtml(line, cite, footnote)); continue; }
    flushTable();
    paragraph.push(line.trim());
  }
  flushParagraph(); flushList(); flushTable(); flushCode(); flushEquation(); flushRawBlock();
  return output.join("\n");
}

function normaliseHtml(source, cite, footnote) {
  return source
    .replace(/<d-contents>[\s\S]*?<\/d-contents>/gi, "")
    .replace(citeTagPattern, (_, attributes) => cite(attributes.match(/\bkey="([^"]+)"/)?.[1] || ""))
    .replace(footnoteTagPattern, (_, note) => footnote(normaliseHtml(note, cite, footnote)))
    .replace(/\s+onerror="[^"]*"/gi, "")
    .replace(/\{\{\s*['"]([^'"]+)['"]\s*\|\s*relative_url\s*\}\}/g, "$1");
}

function parseBibliography(source) {
  const entries = new Map();
  const entryPattern = /@\w+\s*\{\s*([^,]+),([\s\S]*?)(?=\n@|\s*$)/g;
  for (const match of source.matchAll(entryPattern)) {
    const fields = {};
    for (const field of match[2].matchAll(/(author|title|year|url|doi)\s*=\s*[{"]([\s\S]*?)[}"]/gi)) {
      fields[field[1].toLowerCase()] = field[2].replace(/[{}]/g, "").replace(/\s+/g, " ").trim();
    }
    entries.set(match[1].trim(), fields);
  }
  return entries;
}

function authorLabel(author = "") {
  if (!author) return "Unknown author";
  const names = author.split(/\s+and\s+/i).filter(Boolean);
  const first = names[0].trim();
  const surname = first.includes(",") ? first.split(",")[0].trim() : first.split(/\s+/).at(-1);
  return names.length > 1 ? `${surname} et al.` : surname;
}

function citationKeys(source) {
  const keys = [];
  for (const match of source.matchAll(citeTagPattern)) {
    const raw = match[1].match(/\bkey="([^"]+)"/)?.[1] || "";
    for (const key of raw.split(",").map((item) => item.trim()).filter(Boolean)) {
      if (!keys.includes(key)) keys.push(key);
    }
  }
  return keys;
}

function bibliographyHtml(keys, entries) {
  if (!keys.length) return "";
  const references = keys.map((key, index) => {
    const entry = entries.get(key) || {};
    const label = `${authorLabel(entry.author)} (${entry.year || "n.d."}).`;
    const title = entry.title ? ` <em>${escapeHtml(entry.title)}</em>.` : "";
    const destination = entry.url || (entry.doi ? `https://doi.org/${entry.doi}` : "");
    const source = destination ? ` <a href="${escapeAttribute(destination)}" target="_blank" rel="noreferrer">Open source &nearr;</a>` : "";
    return `<li id="reference-${index + 1}"><span class="reference-index">[${index + 1}]</span> ${escapeHtml(label)}${title}${source}</li>`;
  }).join("");
  return `<section class="references"><h2>References</h2><ol>${references}</ol></section>`;
}

function footnotesHtml(notes) {
  if (!notes.length) return "";
  return `<section class="footnotes"><h2>Footnotes</h2><ol>${notes.map((note, index) => `<li id="footnote-${index + 1}">${note}</li>`).join("")}</ol></section>`;
}

function headingText(value) {
  return value.replace(/<[^>]*>/g, "").replace(/&(?:amp|lt|gt|quot|#39);/g, " ").replace(/\s+/g, " ").trim();
}

function articleContents(body) {
  const used = new Set();
  const entries = [];
  const withIds = body.replace(/<h([2-4])([^>]*)>([\s\S]*?)<\/h\1>/gi, (_, level, attributes, content) => {
    const text = headingText(content);
    const currentId = attributes.match(/\bid="([^"]+)"/i)?.[1];
    const base = (currentId || text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "section");
    let id = base;
    let suffix = 2;
    while (used.has(id)) id = `${base}-${suffix++}`;
    used.add(id);
    const nextAttributes = currentId
      ? attributes.replace(/\bid="[^"]+"/i, ` id="${id}"`)
      : `${attributes} id="${id}"`;
    entries.push({ level: Number(level), id, text });
    return `<h${level}${nextAttributes}>${content}</h${level}>`;
  });
  if (!entries.length) return { body: withIds, toc: "" };
  const counts = { 2: 0, 3: 0, 4: 0 };
  const links = entries.map((entry) => {
    if (entry.level === 2) {
      counts[2] += 1;
      counts[3] = 0;
      counts[4] = 0;
    } else if (entry.level === 3) {
      counts[3] += 1;
      counts[4] = 0;
    } else {
      counts[4] += 1;
    }
    const primary = String(Math.max(counts[2], 1)).padStart(2, "0");
    const number = entry.level === 2 ? primary : entry.level === 3 ? `${primary}.${String(counts[3]).padStart(2, "0")}` : `${primary}.${String(Math.max(counts[3], 1)).padStart(2, "0")}.${String(counts[4]).padStart(2, "0")}`;
    return `<a class="${entry.level > 2 ? "toc-subsection" : ""}" href="#${escapeAttribute(entry.id)}"><span>${number}</span> ${escapeHtml(entry.text)}</a>`;
  }).join("");
  return { body: withIds, toc: `<aside class="ledger-article-toc" aria-label="Contents"><span>Contents</span>${links}</aside>` };
}

function staticCorrelationChart(seriesId, title) {
  const series = correlationData.series.find((item) => item.id === seriesId);
  if (!series) return "";
  const points = series.points;
  const width = 760;
  const height = 310;
  const left = 66;
  const right = 28;
  const top = 34;
  const bottom = 52;
  const xMin = points[0].step;
  const xMax = points.at(-1).step || 1;
  const yMin = -1;
  const yMax = 0.2;
  const x = (value) => left + ((value - xMin) / (xMax - xMin || 1)) * (width - left - right);
  const y = (value) => top + ((yMax - value) / (yMax - yMin)) * (height - top - bottom);
  const path = points.map((point) => `${x(point.step).toFixed(1)},${y(point.r).toFixed(1)}`).join(" ");
  const yTicks = [-1, -0.75, -0.5, -0.25, 0];
  const xTicks = seriesId === "Early_localization" ? [0, 2, 4, 6, 8, 10] : [0, 30, 60, 90, 120, 150];
  const grid = yTicks.map((tick) => `<line x1="${left}" x2="${width - right}" y1="${y(tick)}" y2="${y(tick)}" stroke="#c7c5ba" stroke-width="1" ${tick === 0 ? "stroke-dasharray=\"4 4\"" : ""}/><text x="${left - 10}" y="${y(tick) + 4}" text-anchor="end" fill="#73736b" font-size="11">${tick}</text>`).join("");
  const ticks = xTicks.map((tick) => `<text x="${x(tick)}" y="${height - 25}" text-anchor="middle" fill="#73736b" font-size="11">${tick}</text>`).join("");
  const dots = points.map((point) => `<circle cx="${x(point.step)}" cy="${y(point.r)}" r="3.5" fill="#c7422a"/>`).join("");
  const interactionData = escapeAttribute(JSON.stringify(points.map(({ step, r }) => ({ step, r }))));
  return `<svg class="static-chart interactive-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeAttribute(title)}. Hover or focus the chart to inspect values." tabindex="0" data-chart-points="${interactionData}" data-chart-bounds="${left},${top},${width - right},${height - bottom}"><text x="${left}" y="18" fill="#20211e" font-size="13" font-family="Arial, sans-serif">${escapeHtml(title)}</text>${grid}<polyline fill="none" stroke="#1648b8" stroke-width="3" points="${path}"/>${dots}${ticks}<text x="${width / 2}" y="${height - 6}" text-anchor="middle" fill="#73736b" font-size="11">Training step</text><text x="17" y="${height / 2}" transform="rotate(-90 17 ${height / 2})" text-anchor="middle" fill="#73736b" font-size="11">Pearson correlation</text></svg>`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en", { year: "numeric", month: "long", day: "numeric" }).format(new Date(`${value}T00:00:00`));
}

function staticPage(page, title) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>${escapeHtml(title)} - Israel Adewuyi</title><link rel="stylesheet" href="/assets/ledger/site-common.css?v=${assetVersion}" /><link rel="stylesheet" href="/assets/ledger/style.css?v=${assetVersion}" /></head><body data-variant="ledger" data-page="${page}" data-asset-root="/assets/img/"><a class="skip-link" href="#app">Skip to content</a><div id="app"></div><script src="/assets/ledger/routes.js?v=${assetVersion}"></script><script src="/assets/ledger/posts.js?v=${assetVersion}"></script><script src="/assets/ledger/content.js?v=${assetVersion}"></script><script src="/assets/ledger/site-runtime.js?v=${assetVersion}"></script></body></html>`;
}

function articlePage(post, body, toc, references, footnotes) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><meta name="description" content="${escapeAttribute(post.description)}" /><title>${escapeHtml(post.title)} - Israel Adewuyi</title><link rel="stylesheet" href="/assets/ledger/site-common.css" /><link rel="stylesheet" href="/assets/ledger/style.css" /><link rel="stylesheet" href="/assets/ledger/post.css" /><script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script><script defer src="/assets/ledger/article-charts.js"></script></head><body><a class="skip-link" href="#article">Skip to article</a><div class="ledger-post-shell"><aside class="ledger-post-rail"><a class="ledger-post-name" href="/">Israel Adewuyi</a><span>Research ledger / 2024-present</span><nav aria-label="Primary navigation"><a href="/"><span>01</span>Home</a><a href="/research/"><span>02</span>Research</a><a href="/publications/"><span>03</span>Publications</a><a href="/about/"><span>04</span>About</a><a href="/cv/"><span>05</span>Profile</a></nav><p>Research record<br />${escapeHtml(post.date.replaceAll("-", "."))}</p></aside><main class="ledger-post-main" id="article"><div class="ledger-post-top"><span>Research archive / ${escapeHtml(post.topic)}</span><a href="/research/">Back to research</a></div><article class="ledger-post"><header><p class="ledger-post-prompt">$ open ./research/${escapeHtml(post.slug)}</p><p class="ledger-post-meta">${formatDate(post.date)} / ${escapeHtml(post.description)}</p><h1>${escapeHtml(post.title)}</h1></header><div class="ledger-article-layout">${toc}<div class="ledger-post-body">${body}${footnotes}${references}</div></div><footer><span>Research record / ${escapeHtml(post.date.slice(0, 4))}</span><a href="mailto:isistickz@gmail.com">Discuss this note &nearr;</a></footer></article></main></div></body></html>`;
}

const entries = await readdir(postsDirectory, { withFileTypes: true });
const posts = [];
for (const entry of entries) {
  if (!entry.isFile() || !/\.(md|html)$/.test(entry.name)) continue;
  const name = basename(entry.name, extname(entry.name));
  const nameMatch = name.match(/^(\d{4}-\d{2}-\d{2})-(.+)$/);
  if (!nameMatch) continue;
  const source = (await readFile(resolve(postsDirectory, entry.name), "utf8")).replace(/^\uFEFF/, "");
  const { fields, body } = parseFrontMatter(source);
  if (fields.ledger_published !== "true") continue;
  const slug = nameMatch[2];
  const details = existingPostDetails[slug] || {};
  posts.push({
    slug,
    date: fields.date || nameMatch[1],
    topic: fields.topic || details.topic || "Research note",
    title: fields.title || slug.replace(/[-_]/g, " "),
    description: fields.description || "",
    image: fields.image || details.image || "",
    caption: fields.caption || details.caption || "",
    bibliography: fields.bibliography || "",
    sourceType: extname(entry.name),
    body
  });
}

posts.sort((left, right) => right.date.localeCompare(left.date));
await mkdir(ledgerDirectory, { recursive: true });
await writeFile(resolve(ledgerDirectory, "posts.js"), `window.ledgerPosts = ${JSON.stringify(posts.map(({ body, sourceType, ...post }) => post), null, 2)};\n`);
await writeFile(resolve(ledgerDirectory, "routes.js"), `window.siteRoutes = ${JSON.stringify(pageRoutes, null, 2)};\nwindow.siteArticleRoutes = ${JSON.stringify(Object.fromEntries(posts.map((post) => [post.slug, `/blog/${post.date.slice(0, 4)}/${post.slug}/`])), null, 2)};\n`);

for (const [path, page, title] of [["index.html", "home", "Israel Adewuyi"], ["research/index.html", "research", "Research"], ["writing/index.html", "research", "Research"], ["publications/index.html", "publications", "Publications"], ["about/index.html", "about", "About"], ["cv/index.html", "cv", "Profile"], ["blog/index.html", "research", "Research"]]) {
  const destination = resolve(root, path);
  await mkdir(resolve(destination, ".."), { recursive: true });
  await writeFile(destination, staticPage(page, title));
}

await rm(outputDirectory, { recursive: true, force: true });
await mkdir(outputDirectory, { recursive: true });
await writeFile(resolve(outputDirectory, "index.html"), staticPage("research", "Research"));
for (const post of posts) {
  const destination = resolve(outputDirectory, post.date.slice(0, 4), post.slug, "index.html");
  await mkdir(resolve(destination, ".."), { recursive: true });
  const bibliographyPath = post.bibliography ? resolve(root, "assets", "bibliography", post.bibliography) : "";
  const bibliography = bibliographyPath ? parseBibliography(await readFile(bibliographyPath, "utf8").catch(() => "")) : new Map();
  const keys = citationKeys(post.body);
  const index = new Map(keys.map((key, position) => [key, position + 1]));
  const notes = [];
  const cite = (raw) => {
    const values = raw.split(",").map((key) => key.trim()).filter(Boolean);
    return values.length ? `[${values.map((key) => `<a class="citation" href="#reference-${index.get(key) || "?"}">${index.get(key) || "?"}</a>`).join(", ")}]` : "";
  };
  const footnote = (note) => {
    const number = notes.push(note.trim());
    return `<sup class="footnote-ref"><a href="#footnote-${number}" id="footnote-ref-${number}">${number}</a></sup>`;
  };
  let fullBody = post.sourceType === ".html" ? normaliseHtml(post.body, cite, footnote) : renderMarkdown(post.body, cite, footnote);
  if (post.slug === "rlvr-reward-landscape") {
    fullBody = fullBody
      .replace(/<div id="figure-1i-t1-line"[^>]*><\/div>/i, staticCorrelationChart("KL_VS_Reward", "KL vs. reward across training"))
      .replace(/<div id="figure-1ii-t2-line"[^>]*><\/div>/i, staticCorrelationChart("Early_localization", "KL vs. reward during the first 10 steps"));
  }
  const contents = articleContents(fullBody);
  await writeFile(destination, articlePage(post, contents.body, contents.toc, bibliographyHtml(keys, bibliography), footnotesHtml(notes)));
}

console.log(`Built ${posts.length} Ledger posts.`);
