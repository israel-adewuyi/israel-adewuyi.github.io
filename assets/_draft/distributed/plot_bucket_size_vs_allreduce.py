"""Plot bucket size against the number of AllReduce calls.

The built-in measurements are used when neither --data nor --point is passed.
Run ``python plot_bucket_size_vs_allreduce.py --help`` for every option.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DEFAULT_POINTS = [
    (1.0, 80.0),
    (4.0, 53.0),
    (16.0, 47.0),
    (64.0, 24.0),
    (256.0, 8.0),
]

PAPER = "#eeeee8"
INK = "#171916"
MUTED = "#767a73"
HAIRLINE = "#b9bcb5"
BLUE = "#4d91c7"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a configurable x-y line plot with optional point labels and caption.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data",
        type=Path,
        help="CSV file containing the x and y columns.",
    )
    data_group.add_argument(
        "--point",
        action="append",
        metavar="X:Y",
        help="Data point; repeat the option for multiple points, e.g. --point 1:80 --point 4:53.",
    )
    parser.add_argument("--x-column", default="bucket_size_mb", help="CSV column used for x values.")
    parser.add_argument("--y-column", default="all_reduce_calls", help="CSV column used for y values.")
    parser.add_argument("--title", default="Bucket size vs. AllReduce calls")
    parser.add_argument("--subtitle", default="", help="Optional line rendered directly below the title.")
    parser.add_argument("--caption", default="", help="Optional caption rendered below the plot.")
    parser.add_argument("--x-label", default="Bucket size (MB)")
    parser.add_argument("--y-label", default="Number of AllReduce calls")
    parser.add_argument(
        "--x-scale",
        choices=("linear", "log2", "log10"),
        default="log2",
        help="Use log2 for geometrically spaced bucket sizes.",
    )
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--line-color", default=BLUE)
    parser.add_argument("--marker-color", default=BLUE)
    parser.add_argument("--background", default=PAPER)
    parser.add_argument("--grid-color", default=HAIRLINE)
    parser.add_argument("--width", type=float, default=8.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=4.8, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("bucket-size-vs-all-reduce-calls.png"),
        help="Destination; extension selects PNG, SVG, PDF, etc.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Optional destination for a dependency-free interactive HTML version.",
    )
    parser.add_argument(
        "--show-values",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate every point with its y value.",
    )
    parser.add_argument(
        "--grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show quiet horizontal guide lines.",
    )
    parser.add_argument("--transparent", action="store_true", help="Export with a transparent background.")
    return parser


def parse_point(raw: str) -> tuple[float, float]:
    try:
        x_raw, y_raw = raw.split(":", maxsplit=1)
        return float(x_raw), float(y_raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected X:Y, received {raw!r}") from error


def load_csv(path: Path, x_column: str, y_column: str) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or x_column not in reader.fieldnames or y_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or []) or "none"
            raise ValueError(
                f"CSV must contain {x_column!r} and {y_column!r}; available columns: {available}."
            )
        return [(float(row[x_column]), float(row[y_column])) for row in reader]


def load_points(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.data:
        points = load_csv(args.data, args.x_column, args.y_column)
    elif args.point:
        points = [parse_point(raw) for raw in args.point]
    else:
        points = DEFAULT_POINTS.copy()

    if not points:
        raise ValueError("At least one data point is required.")
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in points):
        raise ValueError("All x and y values must be finite numbers.")
    if args.x_scale != "linear" and any(x <= 0 for x, _ in points):
        raise ValueError("Logarithmic x scales require positive x values.")
    return sorted(points)


def format_number(value: float) -> str:
    return f"{value:g}"


def plot(points: list[tuple[float, float]], args: argparse.Namespace) -> Path:
    x_values, y_values = zip(*points)
    figure, axis = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    figure.patch.set_facecolor(args.background)
    axis.set_facecolor(args.background)

    axis.plot(
        x_values,
        y_values,
        color=args.line_color,
        linewidth=2.2,
        marker="o",
        markersize=7,
        markerfacecolor=args.marker_color,
        markeredgecolor=INK,
        markeredgewidth=0.8,
        zorder=3,
    )

    if args.x_scale == "log2":
        axis.set_xscale("log", base=2)
    elif args.x_scale == "log10":
        axis.set_xscale("log", base=10)

    # Always label the supplied x values directly; this is clearer for sparse measurements.
    axis.set_xticks(x_values, labels=[format_number(value) for value in x_values])
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel(args.x_label, color=INK, labelpad=10)
    axis.set_ylabel(args.y_label, color=INK, labelpad=10)
    axis.tick_params(colors=MUTED, labelsize=9)

    if args.grid:
        axis.grid(axis="y", color=args.grid_color, linewidth=0.8, alpha=0.65)
    axis.grid(axis="x", visible=False)

    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(HAIRLINE)

    if args.y_min is not None or args.y_max is not None:
        lower, upper = axis.get_ylim()
        axis.set_ylim(args.y_min if args.y_min is not None else lower, args.y_max if args.y_max is not None else upper)

    if args.show_values:
        for x_value, y_value in points:
            axis.annotate(
                format_number(y_value),
                (x_value, y_value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK,
                fontsize=9,
            )

    if args.title:
        figure.suptitle(args.title, x=0.105, y=0.965, ha="left", color=INK, fontsize=15, fontweight="normal")
    if args.subtitle:
        figure.text(0.105, 0.905, args.subtitle, ha="left", va="top", color=MUTED, fontsize=9)

    bottom = 0.15
    if args.caption:
        caption = textwrap.fill(args.caption, width=max(50, int(args.width * 13)))
        figure.text(0.105, 0.035, caption, ha="left", va="bottom", color=MUTED, fontsize=8.5)
        bottom = 0.22

    figure.subplots_adjust(left=0.105, right=0.97, top=0.84 if args.subtitle else 0.88, bottom=bottom)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, transparent=args.transparent, facecolor=figure.get_facecolor())
    plt.close(figure)
    return args.output


INTERACTIVE_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__DOCUMENT_TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --paper: __BACKGROUND__;
      --ink: __INK__;
      --muted: __MUTED__;
      --hairline: __GRID_COLOR__;
      --line: __LINE_COLOR__;
      --marker: __MARKER_COLOR__;
    }

    * { box-sizing: border-box; }

    html, body { margin: 0; min-width: 0; background: var(--paper); color: var(--ink); }

    body { padding: clamp(14px, 3vw, 28px) clamp(8px, 2.5vw, 24px) 12px; }

    .chart { width: 100%; max-width: 980px; margin: 0 auto; }

    .chart-header { margin: 0 0 4px clamp(50px, 8.7vw, 82px); }

    h1 { margin: 0; font: 400 clamp(17px, 2.6vw, 22px)/1.25 inherit; }

    .subtitle, .caption {
      margin: 5px 0 0;
      color: var(--muted);
      font-size: clamp(11px, 1.7vw, 13px);
      line-height: 1.45;
    }

    .plot-wrap { position: relative; width: 100%; }

    svg { display: block; width: 100%; height: auto; overflow: visible; }

    .axis, .tick { stroke: var(--hairline); stroke-width: 1; vector-effect: non-scaling-stroke; }
    .grid { stroke: var(--hairline); stroke-width: 1; opacity: 0.58; vector-effect: non-scaling-stroke; }
    .series { fill: none; stroke: var(--line); stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .point { fill: var(--marker); stroke: var(--ink); stroke-width: 1; vector-effect: non-scaling-stroke; cursor: pointer; }
    .point-hit { fill: transparent; cursor: pointer; }
    .point-group:hover .point, .point-group.is-active .point { stroke-width: 2.5; }
    .tick-label, .axis-label, .value-label { fill: var(--muted); font-family: inherit; }
    .tick-label { font-size: 12px; }
    .axis-label { fill: var(--ink); font-size: 13px; }
    .value-label { fill: var(--ink); font-size: 12px; }

    .tooltip {
      position: absolute;
      z-index: 2;
      display: none;
      min-width: 150px;
      padding: 8px 10px;
      border: 1px solid var(--hairline);
      background: var(--ink);
      color: var(--paper);
      font-size: 12px;
      line-height: 1.45;
      pointer-events: none;
      transform: translate(-50%, calc(-100% - 14px));
    }

    .tooltip.is-visible { display: block; }
    .tooltip strong { display: block; font-weight: 400; }
    .tooltip span { opacity: 0.78; }

    .caption { margin-left: clamp(50px, 8.7vw, 82px); }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }

    @media (max-width: 480px) {
      body { padding-left: 2px; padding-right: 2px; }
      .chart-header, .caption { margin-left: 47px; }
      .tick-label, .value-label { font-size: 11px; }
      .axis-label { font-size: 12px; }
    }
  </style>
</head>
<body>
  <main class="chart">
    <header class="chart-header">
      <h1 id="chart-title"></h1>
      <p id="chart-subtitle" class="subtitle" hidden></p>
    </header>
    <div id="plot-wrap" class="plot-wrap">
      <svg id="plot" viewBox="0 0 900 400" role="img" aria-labelledby="svg-title svg-description">
        <title id="svg-title"></title>
        <desc id="svg-description"></desc>
      </svg>
      <div id="tooltip" class="tooltip" role="status" aria-live="polite"></div>
    </div>
    <p id="chart-caption" class="caption" hidden></p>
    <table class="sr-only">
      <caption id="table-caption"></caption>
      <thead><tr><th id="table-x"></th><th id="table-y"></th></tr></thead>
      <tbody id="data-table"></tbody>
    </table>
  </main>

  <script>
    (() => {
      "use strict";

      const config = __CONFIG__;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.getElementById("plot");
      const wrap = document.getElementById("plot-wrap");
      const tooltip = document.getElementById("tooltip");
      const width = 900;
      const height = 400;
      const margin = { top: 24, right: 34, bottom: 65, left: 82 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const title = document.getElementById("chart-title");
      const subtitle = document.getElementById("chart-subtitle");
      const caption = document.getElementById("chart-caption");
      title.textContent = config.title;
      title.hidden = !config.title;
      subtitle.textContent = config.subtitle;
      subtitle.hidden = !config.subtitle;
      caption.textContent = config.caption;
      caption.hidden = !config.caption;

      document.getElementById("svg-title").textContent = config.title || `${config.xLabel} versus ${config.yLabel}`;
      document.getElementById("svg-description").textContent =
        `${config.yLabel} decreases from ${format(config.points[0].y)} at ${format(config.points[0].x)} ` +
        `to ${format(config.points.at(-1).y)} at ${format(config.points.at(-1).x)}.`;
      document.getElementById("table-caption").textContent = config.title || "Chart data";
      document.getElementById("table-x").textContent = config.xLabel;
      document.getElementById("table-y").textContent = config.yLabel;

      const table = document.getElementById("data-table");
      config.points.forEach(({ x, y }) => {
        const row = table.insertRow();
        row.insertCell().textContent = format(x);
        row.insertCell().textContent = format(y);
      });

      function element(name, attributes = {}, text = "") {
        const node = document.createElementNS(NS, name);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        if (text) node.textContent = text;
        return node;
      }

      function format(value) {
        return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
      }

      function niceStep(range, round) {
        const exponent = Math.floor(Math.log10(range));
        const fraction = range / (10 ** exponent);
        let niceFraction;
        if (round) {
          niceFraction = fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10;
        } else {
          niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
        }
        return niceFraction * (10 ** exponent);
      }

      function yScale() {
        const values = config.points.map((point) => point.y);
        let low = config.yMin ?? Math.min(0, ...values);
        let high = config.yMax ?? Math.max(...values);
        if (low === high) high = low + 1;
        const step = niceStep((high - low) / 5, true);
        if (config.yMin === null) low = Math.floor(low / step) * step;
        if (config.yMax === null) high = Math.ceil(high / step) * step;
        const ticks = [];
        for (let value = low; value <= high + step / 2; value += step) ticks.push(value);
        return {
          low,
          high,
          ticks,
          position: (value) => margin.top + innerHeight - ((value - low) / (high - low)) * innerHeight,
        };
      }

      function xTransform(value) {
        if (config.xScale === "log2") return Math.log2(value);
        if (config.xScale === "log10") return Math.log10(value);
        return value;
      }

      const transformedX = config.points.map((point) => xTransform(point.x));
      const xLow = Math.min(...transformedX);
      const xHigh = Math.max(...transformedX);
      const xPosition = (value) => {
        if (xLow === xHigh) return margin.left + innerWidth / 2;
        return margin.left + ((xTransform(value) - xLow) / (xHigh - xLow)) * innerWidth;
      };
      const y = yScale();

      const gridLayer = element("g", { "aria-hidden": "true" });
      const axisLayer = element("g", { "aria-hidden": "true" });
      const dataLayer = element("g");
      svg.append(gridLayer, axisLayer, dataLayer);

      if (config.grid) {
        y.ticks.forEach((value) => {
          const py = y.position(value);
          gridLayer.append(element("line", { class: "grid", x1: margin.left, y1: py, x2: width - margin.right, y2: py }));
        });
      }

      axisLayer.append(
        element("line", { class: "axis", x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom }),
        element("line", { class: "axis", x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom })
      );

      y.ticks.forEach((value) => {
        const py = y.position(value);
        axisLayer.append(
          element("line", { class: "tick", x1: margin.left - 5, y1: py, x2: margin.left, y2: py }),
          element("text", { class: "tick-label", x: margin.left - 10, y: py + 4, "text-anchor": "end" }, format(value))
        );
      });

      config.points.forEach((point) => {
        const px = xPosition(point.x);
        axisLayer.append(
          element("line", { class: "tick", x1: px, y1: height - margin.bottom, x2: px, y2: height - margin.bottom + 5 }),
          element("text", { class: "tick-label", x: px, y: height - margin.bottom + 24, "text-anchor": "middle" }, format(point.x))
        );
      });

      axisLayer.append(
        element("text", { class: "axis-label", x: margin.left + innerWidth / 2, y: height - 16, "text-anchor": "middle" }, config.xLabel),
        element("text", {
          class: "axis-label",
          x: 19,
          y: margin.top + innerHeight / 2,
          transform: `rotate(-90 19 ${margin.top + innerHeight / 2})`,
          "text-anchor": "middle",
        }, config.yLabel)
      );

      const path = config.points.map((point, index) => {
        const command = index === 0 ? "M" : "L";
        return `${command}${xPosition(point.x)},${y.position(point.y)}`;
      }).join(" ");
      dataLayer.append(element("path", { class: "series", d: path, "aria-hidden": "true" }));

      let activeGroup = null;

      function showTooltip(group, point, px, py) {
        if (activeGroup) activeGroup.classList.remove("is-active");
        activeGroup = group;
        activeGroup.classList.add("is-active");
        tooltip.innerHTML = `<strong>${format(point.x)} MB bucket</strong><span>${format(point.y)} AllReduce calls</span>`;
        tooltip.classList.add("is-visible");

        const svgRect = svg.getBoundingClientRect();
        const wrapRect = wrap.getBoundingClientRect();
        const left = (px / width) * svgRect.width + svgRect.left - wrapRect.left;
        const top = (py / height) * svgRect.height + svgRect.top - wrapRect.top;
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;

        requestAnimationFrame(() => {
          const half = tooltip.offsetWidth / 2;
          const clamped = Math.max(half + 4, Math.min(wrap.clientWidth - half - 4, left));
          tooltip.style.left = `${clamped}px`;
        });
      }

      function hideTooltip() {
        if (activeGroup) activeGroup.classList.remove("is-active");
        activeGroup = null;
        tooltip.classList.remove("is-visible");
      }

      config.points.forEach((point) => {
        const px = xPosition(point.x);
        const py = y.position(point.y);
        const group = element("g", {
          class: "point-group",
          role: "img",
          "aria-label": `${format(point.x)} megabytes: ${format(point.y)} AllReduce calls`,
        });
        group.append(
          element("circle", { class: "point-hit", cx: px, cy: py, r: 16 }),
          element("circle", { class: "point", cx: px, cy: py, r: 6 })
        );
        if (config.showValues) {
          group.append(element("text", { class: "value-label", x: px, y: py - 13, "text-anchor": "middle" }, format(point.y)));
        }
        group.addEventListener("pointerenter", () => showTooltip(group, point, px, py));
        group.addEventListener("pointerleave", hideTooltip);
        group.addEventListener("click", () => {
          if (activeGroup === group) hideTooltip();
          else showTooltip(group, point, px, py);
        });
        dataLayer.append(group);
      });

      window.addEventListener("resize", hideTooltip);
      document.addEventListener("pointerdown", (event) => {
        if (!event.target.closest(".point-group")) hideTooltip();
      });
    })();
  </script>
</body>
</html>
'''


def write_interactive_html(
    points: list[tuple[float, float]], args: argparse.Namespace, output: Path
) -> Path:
    config = {
        "points": [{"x": x, "y": y} for x, y in points],
        "title": args.title,
        "subtitle": args.subtitle,
        "caption": args.caption,
        "xLabel": args.x_label,
        "yLabel": args.y_label,
        "xScale": args.x_scale,
        "yMin": args.y_min,
        "yMax": args.y_max,
        "showValues": args.show_values,
        "grid": args.grid,
    }
    config_json = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    # Avoid closing the script early if a configurable label contains HTML-like text.
    config_json = config_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document_title = escape(args.title or f"{args.x_label} versus {args.y_label}")
    replacements = {
        "__DOCUMENT_TITLE__": document_title,
        "__BACKGROUND__": args.background,
        "__INK__": INK,
        "__MUTED__": MUTED,
        "__GRID_COLOR__": args.grid_color,
        "__LINE_COLOR__": args.line_color,
        "__MARKER_COLOR__": args.marker_color,
        "__CONFIG__": config_json,
    }
    html = INTERACTIVE_HTML
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        points = load_points(args)
        output = plot(points, args)
        html_output = write_interactive_html(points, args, args.html_output) if args.html_output else None
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output.resolve())
    if html_output:
        print(html_output.resolve())


if __name__ == "__main__":
    main()
