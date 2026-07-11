(() => {
  const ns = "http://www.w3.org/2000/svg";
  const create = (name, attributes = {}) => {
    const element = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  };

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
      const x = left + ((point.step - points[0].step) / (points.at(-1).step - points[0].step || 1)) * (right - left);
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
          Math.abs(left + ((point.step - points[0].step) / (points.at(-1).step - points[0].step || 1)) * (right - left) - relativeX) <
          Math.abs(left + ((points[closest].step - points[0].step) / (points.at(-1).step - points[0].step || 1)) * (right - left) - relativeX)
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
