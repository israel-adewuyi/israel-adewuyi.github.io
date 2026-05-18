---
layout: distill
title: 'RLVR Reward Landscape'
description: 'zooming into the local reward landscape around a policy'
date: 2026-05-05
bibliography: rlvr_landscape.bib
hero_image_scale: "70%"
chart:
  echarts: true
authors:
  - name: Israel Adewuyi
    affiliations:
      name: Innopolis University

toc:
  - name: tldr
  - name: Eval chart demo
  - name: Introduction
  - name: Methodology
    subsections:
      - name: "Overview"
      - name: "Training checkpoints"
      - name: "Evaluation set construction"
      - name: "Landscape construction"
      - name: "Landscape metrics"
  - name: Takeaways
    subsections:
      - name: "T1: High-reward policies become concentrated near the current policy."
      - name: "T2: The concentration happens early in the training."
      - name: "T3: RLVR increasingly behaves like local exploitation around the current policy rather than broad exploration."
  - name: Limitations
  - name: Acknowledgement
---

{% include figure.liquid
  loading="eager"
  path="assets/img/rlvr_landscape/rlvr_reward_landscape.png"
  figure_class="l-body rlvr-hero-image"
  alt="RLVR reward landscape"
  scale=page.hero_image_scale
%}

## tl,dr
- We study 2-dimensional slices of the parameter space around GRPO checkpoints and compare the local surrogate loss, reward, and KL divergence induced by perturbations of the current policy.
- We find that high-reward policies occupy a small region of the local slice and are strongly concentrated near the current policy in distribution space, as measured by the KL divergence.

This work is a WIP case study. The goal is to probe one concrete setup and surface hypotheses about local reward geometry that could be tested across more models, tasks, perturbation scales, and seeds.


## Introduction
Reinforcement Learning with Verifiable Rewards (RLVR) is often motivated as a way to improve pretrained large language models (LLMs) on specific tasks through trial-and-error <d-cite key="lambert2025tulu3pushingfrontiers, guo2025deepseek"></d-cite>. In practice, reasoning traces are sampled from the model, graded by a verifier, and the resulting reward signal is used to update the policy <d-footnote>In this work, we use policy interchangeably with the model</d-footnote> toward higher-scoring trajectories.

A growing debate asks whether RLVR finds and elicits new reasoning capabilities from the models or whether it simply reallocates probability mass towards correct reasoning trajectories that were already accessible to the base models <d-cite key="davis2025objectivereasoningreinforcementlearning, yue2025doesreinforcementlearningreally, wen2025reinforcementlearningverifiablerewards"></d-cite>. This becomes a question of exploration versus exploitation i.e does RLVR continue to discover useful new policies, or does it rapidly become a local refinement operator around the current policy? In RL for LLMs, this question is inherently local because practical algorithms constrain updates to a trust region around the current policy. These constraints are motivated by stability and monotonic-improvement guarantees, but they also imply that any meaningful analysis of exploration or exploitation should be framed in terms of the reachable set of nearby policies rather than unconstrained policy space.

Most existing analyses study the exploration-exploitation question through output-based evaluations, which reveal what a policy can sample but not how training reshapes the nearby policy landscape. In alignment with this trust region view and inspired by loss landscape visualization methods <d-cite key="li2018visualizinglosslandscapeneural"></d-cite>, we perturb intermediate RLVR checkpoints along 2D slices in parameter space and evaluate the resulting counterfactual policies by rollout reward, surrogate loss, and KL divergence from the original checkpoint. This lets us ask how reward is organized within the local reachable neighborhood of the current policy: do high-reward policies remain spread throughout that neighborhood, or do they instead become concentrated near the current policy in distribution space? Through this lens, we study not only whether RLVR improves reward, but how quickly useful exploration becomes local during training.

## Methodology
### Overview
The experiment has two stages. First, we train [**Qwen2.5-0.5B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) with GRPO on [**Alphabet-sort**](https://app.primeintellect.ai/dashboard/environments/kalomaze/alphabet-sort), saving intermediate checkpoints throughout training. Second, we freeze each checkpoint and probe a local two-dimensional slice of the surrounding parameter space. Each point in this slice defines a counterfactual nearby policy, which we evaluate by surrogate loss, reward, and KL divergence from the original checkpoint policy.

In this experiment, GRPO refers to the clipped token-level policy-gradient surrogate used during training. For a batch $\mathcal{B}$, let $y_{i}$ be ith sampled completion in the group and let $\hat{A}_{i}$ be the group-relative advantage. With

$$
\rho_{i,t}(\theta)
=
\frac{\pi_{\text{train}}(y_{i, t} | x, y_{i < t}; \theta)}{\pi_{\text{infer}}(y_{i, t} | x, y_{i < t}; \theta_{old})}
$$ 

the GRPO objective used here is

$$
J_{\mathrm{GRPO}}(\theta)
=
\sum_{x\sim \mathbb{D}, {y_i}_{i=1}^N \sim \pi_{infer}}
\min\!\left[ \frac{1}{\sum_{i=1}^N |y_i|} \sum_{i=1}^N \sum_{t=1}^{|y_i|}
(\rho_{i}(\theta)\hat{A}_{i},
\operatorname{clip}\!\left(\rho_{i}(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_{i})
\right].
$$

<!-- The implementation minimizes the corresponding loss $\mathcal{L}_{\mathrm{GRPO}}(\theta; \mathcal{B}) = -J_{\mathrm{GRPO}}(\theta; \mathcal{B})$. -->

### Training checkpoints
Let $\theta_t$ denote the model parameters during training at step $t$, and let $\pi_{\theta_t}$ denote the corresponding policy. Using GRPO, we update the policy on the Alphabet-sort training environment, where each prompt asks the model to sort an increasing list of names across multiple turns. At a high level, the update takes the form

$$
g_t = \nabla_{\theta} J_{\mathrm{GRPO}}(\theta_t),
\qquad
\theta_{t+1} = \theta_t + \omega_t g_t,
$$

where $\omega_t$ is the step size.<d-footnote>In practice, we use AdamW rather than a plain gradient step.</d-footnote>

We train for 150 steps and save checkpoints every 10 steps and these checkpoints are then treated as fixed objects during landscape construction.

### Evaluation set construction
For landscape evaluation, we use a fixed prompt set

$$
\mathcal{D} = \{x_i\}_{i=1}^{N},
$$

where $N = 1024$. The prompts are sampled from the Alphabet-sort environment. We use the same $\mathcal{D}$ for every checkpoint so that changes in the measured landscapes reflect changes in the policy neighborhood around $\theta_t$, not changes in the evaluation prompts.

### Landscape construction
To study the local geometry around checkpoint $\theta_t$, we sample two random direction tensors $\delta_t$ and $\eta_t$. Each direction is normalized per weight tensor relative to the corresponding tensor norm in $\theta_t$. For sweep coefficients $(\alpha, \beta)$ on a two-dimensional grid, we define the perturbed parameters

$$
\theta_t^{(\alpha, \beta)} = \theta_t + \alpha \delta_t + \beta \eta_t,
$$

and the associated perturbed policy

$$
\pi_t^{(\alpha, \beta)} := \pi_{\theta_t^{(\alpha, \beta)}}.
$$

We evaluate $\pi_t^{(\alpha, \beta)}$ on a $21 \times 21$ grid with $\alpha, \beta \in [-0.05, 0.05]$, giving 441 perturbed policies per checkpoint. For each checkpoint, the sampled directions are fixed across the full grid, and perturbations are applied to all trainable weights. This defines a local two-dimensional parameter-space slice around $\theta_t$.


### Landscape metrics
For each checkpoint $t$ and each grid point $(\alpha, \beta)$, we evaluate three quantities:

<!-- 1. GRPO surrogate loss $J_{\mathrm{GRPO}}(\theta_t^{(\alpha, \beta)})$ with the current policy in place of $\pi_{\text{infer}}$-->
1. GRPO surrogate objective $J_{\mathrm{GRPO}}(\theta_t^{(\alpha,\beta)})$, evaluated on completions sampled from the unperturbed checkpoint policy $\pi_{\theta_t}$ using prompts from the fixed set $\mathcal{D}$;

2. mean rollout reward

   $$
   R_t(\alpha, \beta) =
   \mathbb{E}_{x \sim \mathcal{D}}
   \left[
     r\!\left(x, \pi_t^{(\alpha, \beta)}\right)
   \right],
   $$

   where $r(x, \pi)$ is the verifier-based reward obtained by rolling out policy $\pi$ on prompt $x$;
3. KL divergence from the checkpoint policy

   $$
   \mathrm{KL}_t(\alpha, \beta) =
   \mathbb{E}_{x \sim \mathcal{D}}
   \left[
     D_{\mathrm{KL}}\!\left(
       \pi_{\theta_t}(\cdot \mid x)
       \,\|\,
       \pi_t^{(\alpha, \beta)}(\cdot \mid x)
     \right)
   \right].
   $$

Together, these measurements define a local loss, reward, and divergence landscape around checkpoint $\theta_t$. Repeating this procedure across training checkpoints lets us track how the local high-reward region evolves over the course of RLVR training.

<style>
  .rlvr-corr-figure {
    margin-top: 2rem;
    margin-bottom: 2.5rem;
    scroll-margin-top: 80px;
  }

  .rlvr-summary-figure {
    margin-top: 1.75rem;
    margin-bottom: 2.75rem;
    scroll-margin-top: 80px;
  }

  .rlvr-corr-chart {
    width: 100%;
    height: 360px;
  }

  .rlvr-corr-figure figcaption,
  .rlvr-summary-figure figcaption {
    color: var(--global-text-color-light);
    text-align: center;
  }
</style>

<script>
  (() => {
    const dataUrl = "{{ '/assets/json/rlvr_landscape/correlations.json' | relative_url }}";
    const figureSpecs = [
      {
        elementId: "figure-1i-t1-line",
        seriesId: "KL_VS_Reward",
        title: "Pearson correlation between reward and KL across training steps",
        type: "line",
        color: "#4137ff",
      },
      {
        elementId: "figure-1ii-t2-line",
        seriesId: "Early_localization",
        title: "T2: first 10 steps",
        type: "line",
        color: "#f97316",
      },
    ];

    let correlationData = null;
    const chartInstances = new Map();

    const formatNumber = (value, digits = 3) => {
      if (!Number.isFinite(value)) return String(value);
      if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
      return value.toFixed(digits);
    };

    const getTheme = () => {
      if (typeof determineComputedTheme === "function") return determineComputedTheme();
      return document.documentElement.getAttribute("data-theme") || "light";
    };

    const buildOption = (data, spec, theme) => {
      const source = data.series.find((series) => series.id === spec.seriesId);
      const isLine = spec.type === "line";
      const isDark = theme === "dark";
      const axisColor = isDark ? "rgba(255, 255, 255, 0.72)" : "rgba(0, 0, 0, 0.72)";
      const splitColor = isDark ? "rgba(255, 255, 255, 0.14)" : "rgba(0, 0, 0, 0.12)";
      const textColor = isDark ? "#f3f4f6" : "#111827";
      const tooltipTextColor = isDark ? "#111827" : "#ffffff";

      return {
        animationDuration: 900,
        animationEasing: "cubicOut",
        color: [spec.color],
        title: {
          text: spec.title,
          left: "center",
          top: 0,
          textStyle: {
            color: textColor,
            fontSize: 14,
            fontWeight: 600,
          },
        },
        grid: {
          left: 46,
          right: 24,
          top: 52,
          bottom: 48,
          containLabel: true,
        },
        tooltip: {
          trigger: "item",
          confine: true,
          backgroundColor: isDark ? "rgba(250, 250, 250, 0.95)" : "rgba(17, 17, 17, 0.92)",
          borderWidth: 0,
          textStyle: {
            color: tooltipTextColor,
            fontSize: 12,
          },
          formatter: (params) => {
            const [step, r, n] = params.value;
            return [
              `<span style="color: ${tooltipTextColor}; font-weight: 700;">${params.seriesName}</span>`,
              `step: ${step}`,
              `Pearson correlation: ${formatNumber(r, 3)}`,
              `n: ${n}`,
            ].join("<br/>");
          },
        },
        xAxis: {
          type: "value",
          name: data.x_label,
          nameLocation: "middle",
          nameGap: 30,
          axisLabel: {
            color: axisColor,
          },
          nameTextStyle: {
            color: axisColor,
          },
          axisLine: {
            lineStyle: {
              color: axisColor,
            },
          },
          splitLine: {
            show: false,
          },
        },
        yAxis: {
          type: "value",
          name: data.y_label,
          scale: true,
          axisLabel: {
            color: axisColor,
          },
          nameTextStyle: {
            color: axisColor,
          },
          axisLine: {
            show: true,
            lineStyle: {
              color: axisColor,
            },
          },
          splitLine: {
            lineStyle: {
              color: splitColor,
              type: "dashed",
            },
          },
        },
        series: [
          {
            name: source.name,
            type: spec.type,
            dimensions: ["step", "pearson_r", "n"],
            encode: {
              x: "step",
              y: "pearson_r",
              tooltip: ["step", "pearson_r", "n"],
            },
            data: source.points.map((point) => [point.step, point.r, point.n]),
            symbolSize: isLine ? 7 : 10,
            showSymbol: true,
            smooth: isLine ? 0.25 : false,
            lineStyle: isLine
              ? {
                  width: 3,
                }
              : undefined,
            itemStyle: {
              color: spec.color,
            },
            emphasis: {
              scale: true,
              focus: "series",
            },
            markLine: {
              silent: true,
              symbol: "none",
              label: {
                show: false,
              },
              lineStyle: {
                color: splitColor,
                type: "solid",
              },
              data: [
                {
                  yAxis: 0,
                },
              ],
            },
          },
        ],
      };
    };

    const renderFigures = async () => {
      if (typeof echarts === "undefined") return;
      if (!correlationData) {
        const response = await fetch(dataUrl);
        correlationData = await response.json();
      }

      const theme = getTheme();
      figureSpecs.forEach((spec) => {
        const element = document.getElementById(spec.elementId);
        if (!element) return;

        if (chartInstances.has(spec.elementId)) {
          chartInstances.get(spec.elementId).dispose();
        }

        const chart = echarts.init(element, theme === "dark" ? "dark-fresh-cut" : undefined);
        chart.setOption(buildOption(correlationData, spec, theme));
        chartInstances.set(spec.elementId, chart);
      });
    };

    document.addEventListener("readystatechange", () => {
      if (document.readyState === "complete") renderFigures();
    });

    window.addEventListener("resize", () => {
      chartInstances.forEach((chart) => chart.resize());
    });

    new MutationObserver(renderFigures).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  })();
</script>

## Takeaways

### T1: High-reward policies become concentrated near the current policy.
At each checkpoint $t$, we measure the KL divergence between the current policy and pertubed policy $\theta_t^{(\alpha, \beta)}$, all 441 of them, as well as the reward on $D$. A useful but relatively trivial baseline is that low-KL policies are always concentrated near the current checkpoint - the KL is measured relative to the current policy, so it makes sense that perturbations that leave the policy distribution almost unchanged naturally sit close to $\pi_{\theta_t}$ in distribution space.

The **nontrivial observation** is that reward becomes concentrated around the current checkpoint too! High-reward perturbed policies are not spread uniformly across the local slice. Instead, they tend to appear in the same region where the perturbed policy remains close to the checkpoint distribution.

In <a href="#fig-reward-kl-contours-t1">Figure 1</a>, reward heatmaps and KL contours are plotted over the local parameter slice. If KL localization were the only effect, we would expect the KL contours to be centered near the checkpoint, but reward could still peak elsewhere. Instead, we observe that the high-reward region increasingly overlaps with the low-KL region.

We further quantify this alignment between reward and KL by measuring the Pearson correlation between both terms across the perturbation grid. A negative KL–reward correlation means that, among locally reachable policies, the policies farther from the checkpoint distribution tend to receive lower reward, while policies closer to the checkpoint tend to receive higher reward.

<a href="#fig-corr-t1">Figure 2</a> makes this point across checkpoints. The KL–reward correlation becomes negative early in training, by step 10, indicating that within the reachable neighborhood, moving farther away from the current policy tends to hurt reward.

<figure id="fig-reward-kl-contours-t1" class="l-page rlvr-corr-figure">
  <img
    src="{{ '/assets/img/rlvr_landscape/alphabet_sort_reward_kl_contours_s0_s1_s150_local_scale.png' | relative_url }}"
    alt="Reward and KL contour slices for Alphabet-sort checkpoints at steps 0, 1, and 150"
    width="100%"
  >
  <figcaption>
    <strong>Figure 1.</strong> Reward and KL landscape slices around checkpoints at steps 0, 1, and 150. Each panel is a local two-dimensional parameter-space slice around $\theta_t$, with $(0, 0)$ corresponding to the unperturbed checkpoint. Color denotes mean rollout reward on $\mathcal{D}$, while contour lines denote KL divergence from the checkpoint policy. Each panel uses a local color scale.
  </figcaption>
</figure>

The above observation is interesting as it opens up the question of whether exploration is meant to produce trajectories that are far or close in distribution space. We explore this further in T3.

Which policy the optimizer chooses is another question. 


<figure id="fig-corr-t1" class="l-page rlvr-corr-figure">
  <div id="figure-1i-t1-line" class="rlvr-corr-chart"></div>
  <figcaption>
    <strong>Figure 2.</strong> Pearson correlation between KL divergence and reward across training checkpoints.
  </figcaption>
</figure>

### T2: The concentration happens early in the training.
One possible explanation for T1 is that the concentration is just a late-training effect: once the model has already improved, most high-reward perturbations would naturally be small deviations from the current policy. Under this view, the negative KL-reward correlation would mainly appear after the reward landscape has already settled around a strong checkpoint.

This makes the timing important. If localization only appears near the end of training, it would look more like a consequence of convergence. If it appears in the first few updates, then RLVR may be moving the model into a locally favorable region of policy space much earlier than the final reward curve would suggest, analogous to entering a low-loss basin in supervised learning.

To investigate this, we zoom in on the first 10 training steps. <a href="#fig-corr-t2">Figure 3</a> shows that the KL-reward correlation becomes strongly negative almost immediately, suggesting that the local high-reward region becomes concentrated near the checkpoint policy early in training.

<figure id="fig-corr-t2" class="l-page rlvr-corr-figure">
  <div id="figure-1ii-t2-line" class="rlvr-corr-chart"></div>
  <figcaption>
    <strong>Figure 3.</strong> Pearson correlation between KL divergence and reward over the first 10 training steps.
  </figcaption>
</figure>

### T3: RLVR increasingly behaves like local exploitation around the current policy rather than broad exploration.
A natural reading of T1 and T2 is that RLVR quickly becomes local. High-reward perturbations are not spread broadly across the sampled neighborhood, they are concentrated near the checkpoint policy in KL terms.

How does exploration fit into this? A reasonable conclusion is that the notion of broad exploration over the entire policy space isn't supported by GRPO and by extension, it's derivative algorithms. They seem to rather exploit the local reachable low KL neighbourhood around the current policy. 

The evidence seems to suggest that after a small number of updates, useful exploration may be constrained to a low-KL neighborhood of the current policy. In that sense, RLVR may behave less like broad exploration over policy space and more like exploitation within a locally reachable region.



## Limitations
- **2D slices are lossy views of a high-dimensional object.** The landscape is a 2-dimensional slice though a ~490M-dimensional space. This is useful for probing local structure, but it is still an approximation as the slice may miss directions where reward, loss, or KL behave differently.

- **The perturbation scale matters.** The coefficients $\alpha$ and $\beta$ control how far we zoom out from the checkpoint. In early experiments, larger ranges such as $\pm 0.1$ pushed most perturbed policies into regions with zero reward. Much smaller ranges kept policies too close to the checkpoint, producing too little variation in reward and KL. The range $[-0.05, 0.05]$ gave the most useful resolution for this setup, but this scale is a methodological choice that could be tuned further.

- **The model and task scope are limited.** All results here use Qwen2.5-0.5B-Instruct on Alphabet-sort. The task was chosen because it is simple enough for this model and reaches useful reward levels with relatively few training tokens, making the landscape evaluation computationally feasible. Its partial-credit reward also matters: although rewards lie between 0 and 1, they are not strictly binary, so the landscape still contains useful variation.

The above factors interact. In early GSM8K experiments with binary rewards on Qwen2.5-0.5B, the same $\pm 0.05$ perturbation range was too zoomed out: most perturbed policies received zero reward. This suggests that model scale, task difficulty, reward granularity, and perturbation scale may all affect the observed local geometry and exploration-exploitation behavior under RLVR.


## Appendix

### Configuration summary

#### Training configuration

| Component | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Training environment | Alphabet-sort with `min_turns = 2`, `max_turns = 2` |
| RL algorithm | GRPO |
| Optimizer | AdamW |
| Learning rate | `6e-6` |
| Batch size | `512` |
| Rollouts per example | `16` |
| Sequence length | `4096` |
| Sampling max tokens | `128` |
| Mask truncated completions | `false` |
| Advantage type | `grpo` |
| Advantage epsilon | `1e-8` |
| Loss type | `grpo` |
| Clip epsilon | `0.2` |
| KL coefficient | `0.0` |
| Training steps | `150` |
| Eval interval | Every `10` steps |
| Eval examples | `128` |
| Eval rollouts per example | `1` |
| Eval environment | Alphabet-sort with `min_turns = 2`, `max_turns = 2`, `seed = 2001` |
| Eval sampling | `max_tokens = 128`, `temperature = 0.7` |
| Extra logging interval | Every `30` steps |


### Landscape configuration

| Component | Value |
|---|---|
| Evaluation environment | Alphabet-sort |
| Environment args | `min_turns = 2`, `max_turns = 2`, `seed = 12345` |
| Buffer seed | `12345` |
| Batch size | `1024` |
| Rollouts per example | `16` |
| Sampling max tokens | `512` |
| GRPO clip epsilon | `0.2` |
| Grid range | $\alpha, \beta \in [-0.05, 0.05]$ |
| Grid resolution | $21 \times 21$ |
| Metrics | GRPO surrogate loss, rollout reward, KL divergence |

## Acknowledgment
I am super grateful to Andreas Chollet (for sponsoring compute on this), Daniel David and Professor Ivanov for feedback and insightful questions on earlier drafts of this work.

Prime-Intellect for their easy to use RL library and Qwen team for banger models.
