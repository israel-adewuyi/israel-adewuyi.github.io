(() => {
  const data = window.siteContent;
  const page = document.body.dataset.page || "home";
  const variant = document.body.dataset.variant || "margin";
  const assetRoot = document.body.dataset.assetRoot || "../../../assets/img/";
  const routes = window.siteRoutes || {
    home: "index.html", research: "research.html", writing: "writing.html",
    publications: "publications.html", about: "about.html", cv: "cv.html"
  };
  const links = [
    ["Home", routes.home], ["Research", routes.research],
    ["Publications", routes.publications], ["About", routes.about], ["CV", routes.cv]
  ];
  const q = new URLSearchParams(window.location.search);
  const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const date = (value) => new Intl.DateTimeFormat("en", { year: "numeric", month: "short", day: "numeric" }).format(new Date(`${value}T00:00:00`));
  const active = (href) => page === href.replace(".html", "") || (page === "article" && href === "research.html");
  const nav = () => links.map(([label, href], index) => `<a class="nav-link ${active(href) ? "is-active" : ""}" href="${href}"><span>${String(index + 1).padStart(2, "0")}</span>${label}</a>`).join("");
  const articleUrl = (post) => (window.siteArticleRoutes && window.siteArticleRoutes[post.slug]) || `article.html?slug=${encodeURIComponent(post.slug)}`;
  const postRow = (post, extra = "") => `<a class="post-row" href="${articleUrl(post)}"><time>${date(post.date)}</time><div><h3>${escape(post.title)}</h3><p>${escape(post.description)}</p></div><span class="post-row__topic">${escape(post.topic)}</span><span class="post-row__arrow">&nearr;</span>${extra}</a>`;
  const publication = (item) => `<a class="publication" href="${item.url}" target="_blank" rel="noreferrer"><span>${item.year}</span><div><h3>${escape(item.title)}</h3><p>${escape(item.authors)}</p></div><em>${escape(item.type)} &nearr;</em></a>`;
  const figure = (post) => `<figure class="research-figure"><img src="${assetRoot}${post.image}" alt="${escape(post.caption)}" /><figcaption><span>Figure / ${escape(post.topic)}</span><span>${escape(post.caption)}</span></figcaption></figure>`;
  const home = () => {
    const featured = data.posts.slice(0, 3).map((post) => postRow(post)).join("");
    return `<header class="page-hero home-hero"><p class="prompt">$ whoami</p><h1>${escape(data.identity.name)}</h1><div class="hero-foot"><p>${escape(data.identity.intro)}</p></div></header>
      <section class="section current-section"><span class="section-label">00 / Current focus</span><div class="section-body"><ol class="focus-list"><li><span>01</span><div><strong>Reinforcement learning</strong><p>Current methods understandably favor stable optimization, but may leave too much of the policy space unexplored.</p></div></li><li><span>02</span><div><strong>Distributed training</strong><p>Hardware limits should be baseline for both training and inference.</p></div></li><li><span>03</span><div><strong>Mechanistic interpretability</strong><p>We can eventually understand and map the components that produce a model's behavior.</p></div></li><li><span>04</span><div><strong>Representation learning</strong><p>Understanding how information is encoded is necessary for understanding why a model acts as it does.</p></div></li></ol></div></section>
      <section class="section" id="featured"><span class="section-label">01 / Selected work</span><div class="section-body"><div class="post-list">${featured}</div><a class="section-link" href="${routes.research}">View research archive &rarr;</a></div></section>`;
  };
  const research = () => `<header class="page-hero page-hero--compact"><p class="prompt">$ find ./research -type investigation</p><h1>Research archive</h1><p>Experiments, replications, and technical notes on questions I am curious about.</p></header><section class="section"><span class="section-label">01 / Investigations</span><div class="section-body"><div class="post-list post-list--full">${data.posts.map((post) => postRow(post)).join("")}</div></div></section>`;
  const publications = () => `<header class="page-hero page-hero--compact"><p class="prompt">$ cat ./publications.bib</p><h1>Publications</h1></header><section class="section"><span class="section-label">Bibliography / 02 entries</span><div class="section-body"><div class="publication-list publication-list--full">${data.publications.map(publication).join("")}</div></div></section>`;
  const about = () => `<header class="page-hero page-hero--compact"><h1>Coming soon</h1></header>`;
  const cv = () => `<header class="page-hero page-hero--compact"><h1>Coming soon</h1></header>`;
  const article = () => {
    const post = data.posts.find((item) => item.slug === document.body.dataset.slug || item.slug === q.get("slug")) || data.posts[0];
    document.title = `${post.title} - ${data.identity.name}`;
    const sourceNote = document.body.dataset.productionArticle === "true" ? "" : `<div class="article-source"><span>Migration draft note</span><p>This candidate preserves the real article title, date, topic, figures, and core text. The final migration would carry over the full source Markdown and citations when one visual direction is selected.</p><a href="https://israel-adewuyi.github.io/blog/${post.date.slice(0,4)}/${post.slug === "multiple-winning-subnetworks" ? "slim-peft" : post.slug === "attention-sink" ? "attn_sink_evidence" : post.slug === "rl-meets-ner" ? "ner_with_rl" : post.slug}/" target="_blank" rel="noreferrer">Open current full article &nearr;</a></div>`;
    return `<header class="article-hero"><p class="prompt">$ open ./writing/${escape(post.slug)}</p><p class="article-meta">${date(post.date)} / ${escape(post.topic)}</p><h1>${escape(post.title)}</h1><p>${escape(post.description)}</p></header><article class="article"><aside class="article-toc"><span>Contents</span>${post.sections.map((section, index) => `<a href="#section-${index}">${String(index + 1).padStart(2, "0")} ${escape(section.heading)}</a>`).join("")}</aside><div class="article-content">${post.sections.map((section, index) => `<section id="section-${index}"><h2>${escape(section.heading)}</h2>${section.paragraphs.map((paragraph) => `<p>${escape(paragraph)}</p>`).join("")}${section.bullets ? `<ul>${section.bullets.map((bullet) => `<li>${escape(bullet)}</li>`).join("")}</ul>` : ""}${index === 0 ? figure(post) : ""}</section>`).join("")}${sourceNote}</div></article>`;
  };
  const content = { home, research, publications, about, cv, article }[page] || research;
  const homeContact = page === "home" ? `<section class="site-contact"><span>$ open communication_channel</span><a href="mailto:${data.identity.email}">Say hello &nearr;</a></section>` : "";
  const shell = `<div class="site site--${variant}"><aside class="site-rail"><a class="site-mark" href="index.html">IA</a><div class="site-identity"><strong>${escape(data.identity.name)}</strong><span>AI research</span></div><nav class="site-nav" aria-label="Primary navigation">${nav()}</nav></aside><main class="site-main"><div class="site-top" aria-hidden="true"></div>${content()}${homeContact}<footer class="site-footer"><span>&copy; ${new Date().getFullYear()} ${escape(data.identity.name)}</span><span><a href="mailto:${data.identity.email}">${escape(data.identity.email)}</a> / <a href="https://github.com/israel-adewuyi" target="_blank" rel="noreferrer">GitHub</a></span></footer></main></div>`;
  document.getElementById("app").innerHTML = shell;
  const observer = "IntersectionObserver" in window ? new IntersectionObserver((entries, value) => entries.forEach((entry) => { if (entry.isIntersecting) { entry.target.classList.add("is-visible"); value.unobserve(entry.target); } }), { threshold: 0.1 }) : null;
  document.querySelectorAll(".section, .post-row, .publication, .note-list a, .theme-grid article, .bench-note, .report").forEach((item) => { item.classList.add("reveal"); if (observer) observer.observe(item); else item.classList.add("is-visible"); });
})();
