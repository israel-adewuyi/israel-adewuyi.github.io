(() => {
  "use strict";

  if (window.__autoResizeIframesActive) return;
  window.__autoResizeIframesActive = true;

  const iframeSelector = [
    "d-article iframe:not([data-fixed-height])",
    ".ledger-post-body iframe:not([data-fixed-height])",
  ].join(", ");
  const minimumHeight = 120;
  const framePadding = 2;
  const observers = new WeakMap();

  const numericStyle = (style, property) => Number.parseFloat(style[property]) || 0;

  const contentHeight = (doc) => {
    const body = doc.body;
    const view = doc.defaultView;
    if (!body || !view) return 0;

    const bodyRect = body.getBoundingClientRect();
    const bodyStyle = view.getComputedStyle(body);
    const children = Array.from(body.children).filter((child) => {
      const style = view.getComputedStyle(child);
      return style.display !== "none" && style.position !== "fixed";
    });

    if (!children.length) return Math.ceil(bodyRect.height);

    const contentBottom = children.reduce((bottom, child) => {
      const rect = child.getBoundingClientRect();
      const style = view.getComputedStyle(child);
      return Math.max(bottom, rect.bottom - bodyRect.top + numericStyle(style, "marginBottom"));
    }, numericStyle(bodyStyle, "paddingTop"));

    return Math.ceil(contentBottom + numericStyle(bodyStyle, "paddingBottom"));
  };

  const attach = (iframe) => {
    let resizeFrame = 0;

    const connect = () => {
      const previous = observers.get(iframe);
      if (previous && previous.resizeObserver) previous.resizeObserver.disconnect();

      try {
        const frameUrl = new URL(iframe.src, window.location.href);
        if (frameUrl.origin !== window.location.origin) return;

        const doc = iframe.contentDocument;
        if (!doc || !doc.body || !doc.defaultView || doc.URL === "about:blank") return;

        const resize = () => {
          cancelAnimationFrame(resizeFrame);
          resizeFrame = requestAnimationFrame(() => {
            if (iframe.contentDocument !== doc || !doc.body || !doc.defaultView) return;
            const measured = contentHeight(doc);
            if (!measured) return;
            const nextHeight = Math.max(minimumHeight, measured + framePadding);
            if (Math.abs(iframe.getBoundingClientRect().height - nextHeight) > 1) {
              iframe.style.height = `${nextHeight}px`;
            }
          });
        };

        const resizeObserver = new ResizeObserver(resize);
        const observeContent = () => {
          if (iframe.contentDocument !== doc || !doc.body || !doc.defaultView) return;
          resizeObserver.disconnect();
          resizeObserver.observe(doc.body);
          Array.from(doc.body.children).forEach((child) => resizeObserver.observe(child));
          resize();
        };
        observers.set(iframe, { resizeObserver });

        observeContent();
        if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(resize).catch(() => {});
      } catch (error) {
        // Cross-origin and unavailable iframe documents retain their CSS fallback height.
      }
    };

    iframe.addEventListener("load", connect);
    if (iframe.contentDocument && iframe.contentDocument.readyState === "complete") connect();
  };

  const start = () => document.querySelectorAll(iframeSelector).forEach(attach);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
