(() => {
  "use strict";

  const data = window.distributedResults;
  if (!data?.records?.length) return;

  const NS = "http://www.w3.org/2000/svg";
  const metricMeta = {
    throughput: { label: "Tokens per second", unit: "tokens/s", shortUnit: "tok/s", direction: "Higher is better", decimals: 0 },
    step_time: { label: "Step time", unit: "seconds", shortUnit: "s", direction: "Lower is better", decimals: 2 },
    forward_time: { label: "Forward time", unit: "seconds", shortUnit: "s", direction: "Lower is better", decimals: 2 },
    non_forward_time: { label: "Backward time", unit: "seconds", shortUnit: "s", direction: "Lower is better", decimals: 2 },
    first_stage_memory: { label: "First-stage peak memory", unit: "GiB", shortUnit: "GiB", direction: "Lower is better", decimals: 2 },
    middle_stage_memory: { label: "Middle-stage peak memory", unit: "GiB", shortUnit: "GiB", direction: "Lower is better", decimals: 2 },
    last_stage_memory: { label: "Last-stage peak memory", unit: "GiB", shortUnit: "GiB", direction: "Lower is better", decimals: 2 },
  };
  const viewMeta = {
    dp: { strategy: "DP", title: "Data-parallel results" },
    ep: { strategy: "EP", title: "Expert-parallel results" },
    pp: { strategy: "PP", title: "Pipeline-parallel results" },
    "dp-ep": { strategy: "DP + EP", title: "Data + expert parallel results" },
    "dp-pp": { strategy: "DP + PP", title: "Data + pipeline parallel results" },
    "pp-ep": { strategy: "PP + EP", title: "Pipeline + expert parallel results" },
    "three-d": { strategy: "DP + PP + EP", title: "Three-axis result in 8-GPU context", contextual: true },
  };

  const root = document.getElementById("results-viz");
  const title = document.getElementById("results-title");
  const subtitle = document.getElementById("results-subtitle");
  const kicker = document.getElementById("results-kicker");
  const regimeNote = document.getElementById("regime-note");
  const overviewView = document.getElementById("overview-view");
  const detailView = document.getElementById("detail-view");
  const metricSelect = document.getElementById("metric-select");
  const gpuSelect = document.getElementById("gpu-select");
  const workloadNote = document.getElementById("workload-note");
  const directionNote = document.getElementById("direction-note");
  const overviewMethodNote = document.getElementById("overview-method-note");
  const overviewSvg = document.getElementById("overview-plot");
  const overviewWrap = document.getElementById("overview-wrap");
  const overviewTooltip = document.getElementById("overview-tooltip");
  const detailSvg = document.getElementById("detail-plot");
  const detailWrap = document.getElementById("detail-wrap");
  const detailTooltip = document.getElementById("detail-tooltip");
  const detailControls = document.getElementById("detail-controls");
  const detailMetricSelect = document.getElementById("detail-metric-select");
  const detailLegend = document.getElementById("detail-legend");
  const detailNote = document.getElementById("detail-note");

  const el = (name, attrs = {}, text = "") => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    return node;
  };

  const clear = (node) => {
    Array.from(node.children).forEach((child) => {
      if (!["title", "desc"].includes(child.tagName.toLowerCase())) child.remove();
    });
  };

  const strategyClass = (strategy) => `strategy-${strategy.toLowerCase().replaceAll(" + ", "-").replaceAll(" ", "-")}`;
  const formatNumber = (value, metric) => {
    const meta = metricMeta[metric];
    return new Intl.NumberFormat("en", {
      maximumFractionDigits: meta.decimals,
      minimumFractionDigits: meta.decimals,
    }).format(value);
  };
  const formatTokens = (value) => new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(value);

  const niceStep = (span, targetTicks = 5) => {
    const rough = span / Math.max(1, targetTicks);
    const power = 10 ** Math.floor(Math.log10(rough || 1));
    const error = rough / power;
    const factor = error <= 1 ? 1 : error <= 2 ? 2 : error <= 5 ? 5 : 10;
    return factor * power;
  };

  const ticksFromZero = (maxValue, targetTicks = 5) => {
    const step = niceStep(maxValue, targetTicks);
    const end = Math.ceil(maxValue / step) * step;
    const ticks = [];
    for (let value = 0; value <= end + step / 2; value += step) ticks.push(value);
    return { ticks, max: end };
  };

  const paddedDomain = (values, floorAtZero = false) => {
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      const delta = Math.max(Math.abs(min) * 0.12, 1);
      min -= delta;
      max += delta;
    } else {
      const pad = (max - min) * 0.12;
      min -= pad;
      max += pad;
    }
    if (floorAtZero) min = Math.max(0, min);
    return [min, max];
  };

  const linearTicks = (domain, targetTicks = 5) => {
    const step = niceStep(domain[1] - domain[0], targetTicks);
    const start = Math.ceil(domain[0] / step) * step;
    const end = Math.floor(domain[1] / step) * step;
    const ticks = [];
    for (let value = start; value <= end + step / 2; value += step) ticks.push(value);
    return ticks;
  };

  const tooltipPosition = (tooltip, wrap, svg, px, py) => {
    const svgRect = svg.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    const rawLeft = ((px - viewBox.x) / viewBox.width) * svgRect.width + svgRect.left - wrapRect.left;
    const rawTop = ((py - viewBox.y) / viewBox.height) * svgRect.height + svgRect.top - wrapRect.top;
    tooltip.style.left = `${rawLeft}px`;
    tooltip.style.top = `${rawTop}px`;
    requestAnimationFrame(() => {
      const half = tooltip.offsetWidth / 2;
      tooltip.style.left = `${Math.max(half + 5, Math.min(wrap.clientWidth - half - 5, rawLeft))}px`;
    });
  };

  const hideTooltip = (tooltip, svg) => {
    tooltip.classList.remove("is-visible");
    tooltip.setAttribute("aria-hidden", "true");
    svg.querySelectorAll(".datum").forEach((node) => node.classList.remove("is-active"));
  };

  const recordTooltip = (record, metric) => {
    const stats = record.metrics[metric];
    const unit = metricMeta[metric].shortUnit;
    const metricLine = `<span>${metricMeta[metric].label}: ${formatNumber(stats.mean, metric)} ${unit}</span>`;
    const rangeLine = `<span>Middle 50%: ${formatNumber(stats.q1, metric)}–${formatNumber(stats.q3, metric)} ${unit}</span>`;
    return `<strong>${record.strategy} · ${record.config}</strong>${metricLine}${rangeLine}<span>${record.gpus} GPUs · ${record.hardware.gpu} · ${
      record.hardware.experts
    } experts</span><span>${formatTokens(record.tokensPerStep)} tokens/step</span>`;
  };

  const bindTooltip = ({ group, record, tooltip, wrap, svg, px, py, metric }) => {
    const show = () => {
      svg.querySelectorAll(".datum").forEach((node) => node.classList.remove("is-active"));
      group.classList.add("is-active");
      tooltip.innerHTML = recordTooltip(record, metric);
      tooltip.classList.add("is-visible");
      tooltip.setAttribute("aria-hidden", "false");
      tooltipPosition(tooltip, wrap, svg, px, py);
    };
    group.addEventListener("pointerenter", show);
    group.addEventListener("pointerleave", () => {
      if (document.activeElement !== group) hideTooltip(tooltip, svg);
    });
    group.addEventListener("focus", show);
    group.addEventListener("blur", () => hideTooltip(tooltip, svg));
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      if (group.classList.contains("is-active")) hideTooltip(tooltip, svg);
      else show();
    });
  };

  const metricHasData = (record, metric) =>
    Number.isFinite(record.metrics[metric]?.mean) &&
    (metric !== "middle_stage_memory" || record.metrics[metric].mean > 0);

  const configureMetricOptions = (select, records) => {
    Array.from(select.options).forEach((option) => {
      option.disabled = !records.some((record) => metricHasData(record, option.value));
    });
    if (!select.value || select.selectedOptions[0]?.disabled) {
      const fallback = Array.from(select.options).find((option) => !option.disabled);
      if (fallback) select.value = fallback.value;
    }
  };

  const compactConfig = (record, compact, includeStrategy = false) => {
    let label = record.config;
    if (includeStrategy && record.strategy.includes("+")) label = `${record.strategy} · ${record.shortConfig}`;
    if (!compact) return label;
    if (record.strategy === "PP" && record.config.includes("·")) {
      label = `${record.gpus}×PP · ${record.shortConfig}`;
    }
    return label.replaceAll(" · ", "/").replace("AC ", "AC:");
  };

  function renderOverview() {
    hideTooltip(overviewTooltip, overviewSvg);
    const gpus = Number(gpuSelect.value);
    regimeNote.hidden = gpus !== 8;
    const cohort = data.records.filter((record) => record.gpus === gpus);
    configureMetricOptions(metricSelect, cohort);
    const metric = metricSelect.value;
    const meta = metricMeta[metric];
    const records = cohort.filter((record) => metricHasData(record, metric));
    const sample = records[0];
    const width = Math.max(320, Math.round(overviewWrap.clientWidth));
    const compact = width < 560;
    const left = compact ? 154 : 306;
    const right = compact ? 14 : 32;
    const top = 20;
    const rowGap = records.length > 10 ? (compact ? 32 : 34) : compact ? 38 : 42;
    const bottom = 56;
    const height = top + rowGap * records.length + bottom;
    const chartBottom = height - bottom;
    const chartRight = width - right;
    overviewSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    clear(overviewSvg);

    workloadNote.textContent = `${formatTokens(sample.tokensPerStep)} tokens/step`;
    directionNote.textContent = meta.direction;
    overviewMethodNote.textContent = `Dots are means over steps ${sample.firstStep}–${sample.lastStep}; thin ranges show the middle 50% of logged steps, not repeated-run error bounds.`;

    const maximum = Math.max(...records.map((record) => record.metrics[metric].q3)) * 1.08;
    const tickState = ticksFromZero(maximum, compact ? 3 : 5);
    const x = (value) => left + (value / tickState.max) * (chartRight - left);
    const grid = el("g", { "aria-hidden": "true" });
    const marks = el("g");
    overviewSvg.append(grid, marks);

    tickState.ticks.forEach((tick) => {
      const px = x(tick);
      grid.append(
        el("line", { class: "grid-line", x1: px, x2: px, y1: top - 8, y2: chartBottom + 6 }),
        el("text", { class: "tick-label", x: px, y: chartBottom + 25, "text-anchor": "middle" }, formatNumber(tick, metric))
      );
    });
    grid.append(
      el("rect", { class: "chart-frame", x: left, y: top - 8, width: chartRight - left, height: rowGap * records.length + 14 }),
      el("text", { class: "axis-label", x: (left + chartRight) / 2, y: height - 8, "text-anchor": "middle" }, `${meta.label} (${meta.unit})`)
    );

    records.forEach((record, rowIndex) => {
      const stats = record.metrics[metric];
      const py = top + rowGap * (rowIndex + 0.5);
      const px = x(stats.mean);
      const q1 = x(stats.q1);
      const q3 = x(stats.q3);
      grid.append(
        el("text", { class: "row-label", x: left - 10, y: py + 4, "text-anchor": "end" }, compactConfig(record, compact))
      );
      const group = el("g", {
        class: `datum ${strategyClass(record.strategy)}`,
        tabindex: "0",
        role: "graphics-symbol",
        "aria-label": `${record.strategy}, ${record.config}, ${formatNumber(stats.mean, metric)} ${meta.shortUnit}`,
      });
      group.append(
        el("line", { class: "metric-whisker", x1: q1, x2: q3, y1: py, y2: py }),
        el("line", { class: "metric-cap", x1: q1, x2: q1, y1: py - 4, y2: py + 4 }),
        el("line", { class: "metric-cap", x1: q3, x2: q3, y1: py - 4, y2: py + 4 }),
        el("circle", { class: "hit-target", cx: px, cy: py, r: compact ? 17 : 15 }),
        el("circle", { class: "metric-mark", cx: px, cy: py, r: 5.5 })
      );
      bindTooltip({ group, record, tooltip: overviewTooltip, wrap: overviewWrap, svg: overviewSvg, px, py, metric });
      marks.append(group);
    });
  }

  function renderDetail(view) {
    hideTooltip(detailTooltip, detailSvg);
    const meta = viewMeta[view];
    const cohort = meta.contextual
      ? data.records.filter((record) => record.gpus === 8)
      : data.records.filter((record) => record.strategy === meta.strategy);
    regimeNote.hidden = !cohort.some((record) => record.gpus === 8);
    configureMetricOptions(detailMetricSelect, cohort);
    const metric = detailMetricSelect.value;
    const metricInfo = metricMeta[metric];
    const records = cohort.filter((record) => metricHasData(record, metric));
    const width = Math.max(320, Math.round(detailWrap.clientWidth));
    const compact = width < 560;
    const left = compact ? 154 : 306;
    const right = compact ? 14 : 34;
    const top = 18;
    const rowGap = records.length > 10 ? 29 : compact ? 38 : 42;
    const bottom = 56;
    const height = top + rowGap * records.length + bottom;
    const chartRight = width - right;
    const chartBottom = height - bottom;
    detailSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    clear(detailSvg);

    const maximum = Math.max(...records.map((record) => record.metrics[metric].q3)) * 1.08;
    const tickState = ticksFromZero(maximum, compact ? 3 : 5);
    const x = (value) => left + (value / tickState.max) * (chartRight - left);
    const grid = el("g", { "aria-hidden": "true" });
    const marks = el("g");
    detailSvg.append(grid, marks);

    tickState.ticks.forEach((tick) => {
      const px = x(tick);
      grid.append(
        el("line", { class: "grid-line", x1: px, x2: px, y1: top - 8, y2: chartBottom + 6 }),
        el("text", { class: "tick-label", x: px, y: chartBottom + 25, "text-anchor": "middle" }, formatNumber(tick, metric))
      );
    });
    grid.append(
      el("rect", { class: "chart-frame", x: left, y: top - 8, width: chartRight - left, height: rowGap * records.length + 14 }),
      el("text", { class: "axis-label", x: (left + chartRight) / 2, y: height - 8, "text-anchor": "middle" }, `${metricInfo.label} (${metricInfo.unit})`)
    );

    records.forEach((record, rowIndex) => {
      const stats = record.metrics[metric];
      const py = top + rowGap * (rowIndex + 0.5);
      const px = x(stats.mean);
      const q1 = x(stats.q1);
      const q3 = x(stats.q3);
      const contextual = meta.contextual && record.strategy !== meta.strategy;
      const target = meta.contextual && record.strategy === meta.strategy;
      grid.append(
        el("text", { class: "row-label", x: left - 10, y: py + 4, "text-anchor": "end" }, compactConfig(record, compact, meta.contextual))
      );
      const group = el("g", {
        class: `datum ${strategyClass(record.strategy)}${contextual ? " is-context" : ""}${target ? " is-target" : ""}`,
        tabindex: "0",
        role: "graphics-symbol",
        "aria-label": `${record.strategy}, ${record.config}, ${formatNumber(stats.mean, metric)} ${metricInfo.shortUnit}`,
      });
      group.append(
        el("line", { class: "metric-whisker", x1: q1, x2: q3, y1: py, y2: py }),
        el("line", { class: "metric-cap", x1: q1, x2: q1, y1: py - 4, y2: py + 4 }),
        el("line", { class: "metric-cap", x1: q3, x2: q3, y1: py - 4, y2: py + 4 }),
        el("circle", { class: "hit-target", cx: px, cy: py, r: compact ? 18 : 16 }),
        el("circle", { class: "metric-mark", cx: px, cy: py, r: target ? 7 : 5.5 })
      );
      bindTooltip({ group, record, tooltip: detailTooltip, wrap: detailWrap, svg: detailSvg, px, py, metric });
      marks.append(group);
    });

    detailLegend.innerHTML = "";
    detailNote.textContent = "";
  }

  function renderBucketSweep() {
    hideTooltip(detailTooltip, detailSvg);
    const records = data.bucketSweep;
    const width = Math.max(320, Math.round(detailWrap.clientWidth));
    const compact = width < 560;
    const height = compact ? 380 : 420;
    const left = compact ? 58 : 74;
    const right = compact ? 18 : 34;
    const top = 24;
    const bottom = 60;
    const chartRight = width - right;
    const chartBottom = height - bottom;
    detailSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    clear(detailSvg);

    const yDomain = paddedDomain(records.flatMap((record) => [record.deltaNonForwardMs.q1, record.deltaNonForwardMs.q3]));
    yDomain[0] = Math.min(yDomain[0], 0);
    yDomain[1] = Math.max(yDomain[1], 0);
    const x = (index) => left + 14 + (index / Math.max(1, records.length - 1)) * (chartRight - left - 28);
    const y = (value) => chartBottom - ((value - yDomain[0]) / (yDomain[1] - yDomain[0])) * (chartBottom - top);
    const yTicks = linearTicks(yDomain, compact ? 4 : 6);
    const grid = el("g", { "aria-hidden": "true" });
    const marks = el("g");
    detailSvg.append(grid, marks);

    yTicks.forEach((tick) => {
      const py = y(tick);
      grid.append(
        el("line", { class: tick === 0 ? "zero-line" : "grid-line", x1: left, x2: chartRight, y1: py, y2: py }),
        el("text", { class: "tick-label", x: left - 9, y: py + 4, "text-anchor": "end" }, tick.toFixed(0))
      );
    });
    records.forEach((record, index) => {
      const px = x(index);
      grid.append(el("text", { class: "tick-label", x: px, y: chartBottom + 23, "text-anchor": "middle" }, String(record.bucketMiB)));
      const stats = record.deltaNonForwardMs;
      const py = y(stats.mean);
      const group = el("g", {
        class: "datum strategy-dp",
        tabindex: "0",
        role: "graphics-symbol",
        "aria-label": `${record.bucketMiB} MiB bucket: ${stats.mean.toFixed(1)} milliseconds versus the 0 MiB baseline`,
      });
      group.append(
        el("circle", { class: "hit-target", cx: px, cy: py, r: compact ? 20 : 17 }),
        el("circle", { class: "metric-mark", cx: px, cy: py, r: 6 }),
        el("text", { class: "value-label", x: px, y: py - 12, "text-anchor": "middle" }, `${stats.mean > 0 ? "+" : ""}${stats.mean.toFixed(1)}`)
      );
      const show = () => {
        detailSvg.querySelectorAll(".datum").forEach((node) => node.classList.remove("is-active"));
        group.classList.add("is-active");
        detailTooltip.innerHTML = `<strong>${record.bucketMiB} MiB reducer bucket</strong><span>Change vs 0 MiB: ${
          stats.mean > 0 ? "+" : ""
        }${stats.mean.toFixed(2)} ms</span><span>Non-forward time: ${record.nonForwardTime.mean.toFixed(
          3
        )} s</span><span>Throughput: ${record.throughput.mean.toFixed(0)} tokens/s</span>`;
        detailTooltip.classList.add("is-visible");
        detailTooltip.setAttribute("aria-hidden", "false");
        tooltipPosition(detailTooltip, detailWrap, detailSvg, px, py);
      };
      group.addEventListener("pointerenter", show);
      group.addEventListener("pointerleave", () => {
        if (document.activeElement !== group) hideTooltip(detailTooltip, detailSvg);
      });
      group.addEventListener("focus", show);
      group.addEventListener("blur", () => hideTooltip(detailTooltip, detailSvg));
      marks.append(group);
    });
    grid.append(
      el("rect", { class: "chart-frame", x: left, y: top, width: chartRight - left, height: chartBottom - top }),
      el("text", { class: "axis-label", x: (left + chartRight) / 2, y: height - 9, "text-anchor": "middle" }, "Reducer bucket size (MiB)"),
      el(
        "text",
        { class: "axis-label", x: 15, y: (top + chartBottom) / 2, transform: `rotate(-90 15 ${(top + chartBottom) / 2})`, "text-anchor": "middle" },
        "Change in non-forward time (ms)"
      )
    );
    detailLegend.innerHTML = "";
    detailNote.textContent =
      "Values are paired against the 0 MiB run at the same training step. Negative is faster; the effect is small relative to the roughly 5-second non-forward pass.";
  }

  const rollingMean = (points, windowSize = 9) =>
    points.map((point, index) => {
      const start = Math.max(0, index - windowSize + 1);
      const window = points.slice(start, index + 1);
      return {
        tokens: point.tokens,
        loss: window.reduce((sum, candidate) => sum + candidate.loss, 0) / window.length,
      };
    });

  function renderLearningRateSweep() {
    hideTooltip(detailTooltip, detailSvg);
    const series = data.lrSweep.map((item) => ({ ...item, smoothed: rollingMean(item.points) }));
    const width = Math.max(320, Math.round(detailWrap.clientWidth));
    const compact = width < 560;
    const height = compact ? 390 : 430;
    const left = compact ? 58 : 72;
    const right = compact ? 18 : 34;
    const top = 22;
    const bottom = 58;
    const chartRight = width - right;
    const chartBottom = height - bottom;
    detailSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    clear(detailSvg);

    const allPoints = series.flatMap((item) => item.smoothed);
    const xDomain = [0, Math.max(...allPoints.map((point) => point.tokens))];
    const yDomain = paddedDomain(
      allPoints.map((point) => point.loss),
      true
    );
    const x = (value) => left + (value / xDomain[1]) * (chartRight - left);
    const y = (value) => chartBottom - ((value - yDomain[0]) / (yDomain[1] - yDomain[0])) * (chartBottom - top);
    const xTicks = ticksFromZero(xDomain[1], compact ? 3 : 5).ticks.filter((tick) => tick <= xDomain[1]);
    const yTicks = linearTicks(yDomain, compact ? 4 : 6);
    const grid = el("g", { "aria-hidden": "true" });
    const marks = el("g");
    const hover = el("g", { "aria-hidden": "true", visibility: "hidden" });
    const guide = el("line", { class: "hover-guide", y1: top, y2: chartBottom });
    hover.append(guide);
    detailSvg.append(grid, marks, hover);

    xTicks.forEach((tick) => {
      const px = x(tick);
      grid.append(
        el("line", { class: "grid-line", x1: px, x2: px, y1: top, y2: chartBottom }),
        el("text", { class: "tick-label", x: px, y: chartBottom + 23, "text-anchor": "middle" }, (tick / 1_000_000).toFixed(tick ? 1 : 0))
      );
    });
    yTicks.forEach((tick) => {
      const py = y(tick);
      grid.append(
        el("line", { class: "grid-line", x1: left, x2: chartRight, y1: py, y2: py }),
        el("text", { class: "tick-label", x: left - 9, y: py + 4, "text-anchor": "end" }, tick.toFixed(1))
      );
    });
    grid.append(
      el("rect", { class: "chart-frame", x: left, y: top, width: chartRight - left, height: chartBottom - top }),
      el("text", { class: "axis-label", x: (left + chartRight) / 2, y: height - 9, "text-anchor": "middle" }, "Tokens seen (millions)"),
      el(
        "text",
        { class: "axis-label", x: 15, y: (top + chartBottom) / 2, transform: `rotate(-90 15 ${(top + chartBottom) / 2})`, "text-anchor": "middle" },
        "Total loss · 9-step rolling mean"
      )
    );

    series.forEach((item, index) => {
      const path = item.smoothed
        .map((point, pointIndex) => `${pointIndex ? "L" : "M"}${x(point.tokens).toFixed(2)},${y(point.loss).toFixed(2)}`)
        .join(" ");
      const group = el("g", {
        class: `lr-${index + 1}`,
        tabindex: "0",
        role: "graphics-symbol",
        "aria-label": `Learning rate ${item.learningRate}; final smoothed loss ${item.smoothed.at(-1).loss.toFixed(2)}`,
      });
      group.append(el("path", { class: "lr-path", d: path }));
      marks.append(group);
    });

    const overlay = el("rect", { class: "hit-target", x: left, y: top, width: chartRight - left, height: chartBottom - top });
    const showAt = (event) => {
      const rect = detailSvg.getBoundingClientRect();
      const localX = ((event.clientX - rect.left) / rect.width) * width;
      const clampedX = Math.max(left, Math.min(chartRight, localX));
      const tokens = ((clampedX - left) / (chartRight - left)) * xDomain[1];
      const nearest = series.map((item, index) => {
        let best = item.smoothed[0];
        for (const point of item.smoothed) {
          if (Math.abs(point.tokens - tokens) < Math.abs(best.tokens - tokens)) best = point;
        }
        return { item, index, point: best };
      });
      hover.replaceChildren(guide);
      guide.setAttribute("x1", clampedX);
      guide.setAttribute("x2", clampedX);
      nearest.forEach(({ index, point }) =>
        hover.append(el("circle", { class: `hover-marker lr-${index + 1}`, cx: x(point.tokens), cy: y(point.loss), r: 4.5 }))
      );
      hover.setAttribute("visibility", "visible");
      detailTooltip.innerHTML = `<strong>${formatTokens(nearest[0].point.tokens)} tokens seen</strong>${nearest
        .map(({ item, point }) => `<span>LR ${item.learningRate.toExponential(0)}: ${point.loss.toFixed(2)} loss</span>`)
        .join("")}`;
      detailTooltip.classList.add("is-visible");
      detailTooltip.setAttribute("aria-hidden", "false");
      tooltipPosition(detailTooltip, detailWrap, detailSvg, clampedX, Math.min(...nearest.map(({ point }) => y(point.loss))));
    };
    overlay.addEventListener("pointermove", showAt);
    overlay.addEventListener("pointerleave", () => {
      hover.setAttribute("visibility", "hidden");
      hideTooltip(detailTooltip, detailSvg);
    });
    marks.append(overlay);

    detailLegend.innerHTML = series
      .map(
        (item, index) => `<span class="legend-item lr-${index + 1}"><span class="legend-line"></span>LR ${item.learningRate.toExponential(0)}</span>`
      )
      .join("");
    detailNote.textContent =
      "One-GPU DP learning-rate sweep on RTX 3090 with 20 experts. Curves use a trailing 9-step mean to expose the loss trajectory without hiding the underlying run length.";
  }

  function initialise() {
    const view = location.hash.slice(1) || "overview";
    if (view === "overview") {
      overviewView.hidden = false;
      detailView.hidden = true;
      kicker.textContent = "nanoTitan benchmark summary";
      title.textContent = "Parallelism comparison";
      subtitle.textContent =
        "Choose a metric and GPU count. Every point is one logged configuration at the same workload (batch size / step) within that GPU count.";
      renderOverview();
      return;
    }
    if (view === "bucket" || view === "lr") {
      overviewView.hidden = true;
      detailView.hidden = false;
      regimeNote.hidden = true;
      detailControls.hidden = true;
      kicker.textContent = view === "bucket" ? "reducer experiment" : "optimisation sweep";
      title.textContent = view === "bucket" ? "Reducer bucket-size result" : "Data-parallel learning-rate sweep";
      subtitle.textContent =
        view === "bucket"
          ? "Paired changes in non-forward step time relative to the 0 MiB baseline."
          : "Total-loss trajectories for every logged learning-rate configuration.";
      if (view === "bucket") renderBucketSweep();
      else renderLearningRateSweep();
      return;
    }
    if (!viewMeta[view]) {
      location.hash = "overview";
      return;
    }
    overviewView.hidden = true;
    detailView.hidden = false;
    detailControls.hidden = false;
    kicker.textContent = "experiment configurations";
    title.textContent = viewMeta[view].title;
    subtitle.textContent = "Choose a metric to compare every logged configuration in this section.";
    renderDetail(view);
  }

  metricSelect.addEventListener("change", renderOverview);
  gpuSelect.addEventListener("change", renderOverview);
  detailMetricSelect.addEventListener("change", () => {
    const view = location.hash.slice(1);
    if (viewMeta[view]) renderDetail(view);
  });
  window.addEventListener("hashchange", initialise);
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".datum")) {
      hideTooltip(overviewTooltip, overviewSvg);
      hideTooltip(detailTooltip, detailSvg);
    }
  });

  let resizeFrame = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(initialise);
  }).observe(root);

  initialise();
})();
