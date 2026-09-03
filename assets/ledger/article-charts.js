(() => {
  const loadIframeResizer = () => {
    if (!document.querySelector(".ledger-post-body iframe")) return;
    if (document.querySelector('script[data-auto-resize-iframes]')) return;

    const script = document.createElement("script");
    script.src = "/assets/js/auto-resize-iframes.js";
    script.dataset.autoResizeIframes = "";
    document.head.append(script);
  };

  loadIframeResizer();

  const ns = "http://www.w3.org/2000/svg";
  const create = (name, attributes = {}) => {
    const element = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  };

  const setupArticleToc = () => {
    const toc = document.querySelector(".ledger-article-toc");
    if (!toc) return;

    let linksContainer = toc.querySelector(".toc-links");
    if (!linksContainer) {
      linksContainer = document.createElement("div");
      linksContainer.className = "toc-links";
      linksContainer.id = "article-toc-links";
      Array.from(toc.children)
        .filter((child) => child.tagName === "A")
        .forEach((link) => linksContainer.append(link));
      toc.append(linksContainer);
    }

    let toggle = toc.querySelector(".toc-mobile-toggle");
    if (!toggle) {
      toggle = document.createElement("button");
      toggle.className = "toc-mobile-toggle";
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-controls", linksContainer.id);
      const label = document.createElement("span");
      label.className = "toc-mobile-label";
      label.textContent = "Contents";
      const icon = document.createElement("span");
      icon.className = "toc-mobile-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "\u25be";
      toggle.append(label, icon);
      toc.insertBefore(toggle, linksContainer);
    }

    const mobileLabel = toggle.querySelector(".toc-mobile-label");
    const mobileQuery = window.matchMedia("(max-width: 900px)");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const links = Array.from(linksContainer.querySelectorAll('a[href^="#"]'));

    links.forEach((link) => {
      if (!link.classList.contains("toc-subsection")) return;
      if (link.classList.contains("toc-level-3") || link.classList.contains("toc-level-4")) return;
      const markerNode = link.querySelector("span");
      const marker = markerNode ? markerNode.textContent : "";
      const depth = (marker.match(/\./g) || []).length;
      link.classList.add(depth >= 2 ? "toc-level-4" : "toc-level-3");
    });

    const entries = links
      .map((link) => {
        const id = decodeURIComponent(link.hash.slice(1));
        const heading = document.getElementById(id);
        return heading ? { heading, link } : null;
      })
      .filter(Boolean);
    if (!entries.length) return;

    const linkLabel = (link) => {
      const copy = link.cloneNode(true);
      const marker = copy.querySelector("span");
      if (marker) marker.remove();
      return copy.textContent.trim();
    };
    const setOpen = (open) => {
      toc.classList.toggle("is-open", open);
      toggle.setAttribute("aria-expanded", String(open));
    };
    const keepLinkVisible = (link) => {
      if (mobileQuery.matches && !toc.classList.contains("is-open")) return;
      if (linksContainer.scrollHeight <= linksContainer.clientHeight) return;
      const containerRect = linksContainer.getBoundingClientRect();
      const linkRect = link.getBoundingClientRect();
      let nextTop = linksContainer.scrollTop;
      if (linkRect.top < containerRect.top + 8) nextTop += linkRect.top - containerRect.top - 8;
      if (linkRect.bottom > containerRect.bottom - 8) nextTop += linkRect.bottom - containerRect.bottom + 8;
      if (nextTop === linksContainer.scrollTop) return;
      linksContainer.scrollTo({ top: nextTop, behavior: reducedMotion.matches ? "auto" : "smooth" });
    };

    let activeLink = null;
    const setActive = (link) => {
      if (!link || link === activeLink) return;
      links.forEach((item) => item.removeAttribute("aria-current"));
      link.setAttribute("aria-current", "location");
      activeLink = link;
      if (mobileLabel) mobileLabel.textContent = `Contents \u00b7 ${linkLabel(link)}`;
      requestAnimationFrame(() => keepLinkVisible(link));
    };
    const updateActive = () => {
      const threshold = mobileQuery.matches ? 124 : 72;
      let current = entries[0];
      for (const entry of entries) {
        if (entry.heading.getBoundingClientRect().top > threshold) break;
        current = entry;
      }
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) current = entries[entries.length - 1];
      setActive(current.link);
    };

    let framePending = false;
    const scheduleUpdate = () => {
      if (framePending) return;
      framePending = true;
      requestAnimationFrame(() => {
        framePending = false;
        updateActive();
      });
    };

    toggle.addEventListener("click", () => {
      const open = !toc.classList.contains("is-open");
      setOpen(open);
      if (open && activeLink) requestAnimationFrame(() => keepLinkVisible(activeLink));
    });
    links.forEach((link) => link.addEventListener("click", () => {
      if (mobileQuery.matches) setOpen(false);
    }));
    const updateMode = () => {
      setOpen(false);
      scheduleUpdate();
    };
    if (mobileQuery.addEventListener) mobileQuery.addEventListener("change", updateMode);
    else mobileQuery.addListener(updateMode);
    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("hashchange", scheduleUpdate);
    updateActive();
  };

  setupArticleToc();

  document.querySelectorAll(".interactive-chart[data-chart-points]").forEach((chart) => {
    const points = JSON.parse(chart.dataset.chartPoints || "[]");
    const [left, top, right, bottom] = (chart.dataset.chartBounds || "").split(",").map(Number);
    if (!points.length || [left, top, right, bottom].some(Number.isNaN)) return;

    const cursor = create("line", { stroke: "#1648b8", "stroke-width": "1", "stroke-dasharray": "4 4", visibility: "hidden" });
    const marker = create("circle", { r: "6", fill: "#c7422a", stroke: "#f6f4ed", "stroke-width": "3", visibility: "hidden" });
    const label = create("g", { visibility: "hidden", "pointer-events": "none" });
    const labelBox = create("rect", { width: "138", height: "42", rx: "2", fill: "#20211e" });
    const labelText = create("text", { x: "10", y: "17", fill: "#ffffff", "font-size": "11", "font-family": "Consolas, monospace" });
    const labelValue = create("text", { x: "10", y: "33", fill: "#d9e3ff", "font-size": "12", "font-family": "Consolas, monospace" });
    label.append(labelBox, labelText, labelValue);
    chart.append(cursor, marker, label);

    const show = (index) => {
      const point = points[index];
      const x = left + ((point.step - points[0].step) / (points[points.length - 1].step - points[0].step || 1)) * (right - left);
      const y = top + ((0.2 - point.r) / 1.2) * (bottom - top);
      const labelX = x > right - 160 ? x - 150 : x + 12;
      const labelY = Math.max(top + 6, Math.min(bottom - 48, y - 48));
      cursor.setAttribute("x1", x);
      cursor.setAttribute("x2", x);
      cursor.setAttribute("y1", top);
      cursor.setAttribute("y2", bottom);
      marker.setAttribute("cx", x);
      marker.setAttribute("cy", y);
      label.setAttribute("transform", `translate(${labelX} ${labelY})`);
      labelText.textContent = `STEP ${point.step}`;
      labelValue.textContent = `r = ${point.r.toFixed(3)}`;
      [cursor, marker, label].forEach((element) => element.setAttribute("visibility", "visible"));
      chart.setAttribute("aria-label", `Training step ${point.step}; Pearson correlation ${point.r.toFixed(3)}.`);
      chart.dataset.chartIndex = String(index);
    };

    const hide = () => [cursor, marker, label].forEach((element) => element.setAttribute("visibility", "hidden"));
    const nearest = (event) => {
      const rectangle = chart.getBoundingClientRect();
      const relativeX = ((event.clientX - rectangle.left) / rectangle.width) * chart.viewBox.baseVal.width;
      return points.reduce(
        (closest, point, index) =>
          Math.abs(
            left +
              ((point.step - points[0].step) / (points[points.length - 1].step - points[0].step || 1)) * (right - left) -
              relativeX
          ) <
          Math.abs(
            left +
              ((points[closest].step - points[0].step) / (points[points.length - 1].step - points[0].step || 1)) *
                (right - left) -
              relativeX
          )
            ? index
            : closest,
        0
      );
    };

    chart.addEventListener("pointermove", (event) => show(nearest(event)));
    chart.addEventListener("pointerleave", hide);
    chart.addEventListener("focus", () => show(Number(chart.dataset.chartIndex || 0)));
    chart.addEventListener("blur", hide);
    chart.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const current = Number(chart.dataset.chartIndex || 0);
      show(Math.max(0, Math.min(points.length - 1, current + (event.key === "ArrowRight" ? 1 : -1))));
    });
  });
})();
