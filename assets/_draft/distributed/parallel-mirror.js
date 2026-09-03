(function () {
  const root = document.querySelector(".parallel-draft");
  if (!root) return;

  const buttons = Array.from(root.querySelectorAll("[data-phase-target]"));
  const copy = root.querySelector("[data-phase-copy]");
  const descriptions = {
    forward: root.dataset.forwardCopy || "",
    backward: root.dataset.backwardCopy || "",
  };

  function setPhase(phase) {
    if (phase !== "forward" && phase !== "backward") return;
    root.dataset.phase = phase;
    buttons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.phaseTarget === phase));
    });
    if (copy) copy.innerHTML = descriptions[phase];
    document.title = `${root.dataset.draftName || "Data parallelism"} · ${phase}`;
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => setPhase(button.dataset.phaseTarget));
  });

  const requestedPhase = new URLSearchParams(window.location.search).get("phase");
  setPhase(requestedPhase === "backward" ? "backward" : "forward");
  window.setPhase = setPhase;
})();
