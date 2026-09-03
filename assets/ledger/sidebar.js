(() => {
  const mobile = window.matchMedia("(max-width: 900px)");
  const storageKey = "ledger-sidebar-collapsed";

  const readCollapsed = () => {
    try {
      return window.localStorage.getItem(storageKey) === "true";
    } catch (error) {
      return false;
    }
  };

  const writeCollapsed = (value) => {
    try {
      window.localStorage.setItem(storageKey, String(value));
    } catch (error) {
      // The sidebar still works when storage is unavailable.
    }
  };

  document.querySelectorAll("[data-sidebar-shell]").forEach((shell) => {
    const button = shell.querySelector("[data-sidebar-toggle]");
    const buttonText = button ? button.querySelector("[data-sidebar-toggle-text]") : null;
    const backdrop = shell.querySelector("[data-sidebar-backdrop]");
    const navigation = shell.querySelector("[data-sidebar-nav]");
    if (!button || !backdrop || !navigation) return;

    const setMobileOpen = (open) => {
      shell.classList.toggle("is-sidebar-open", open);
      backdrop.hidden = !open;
      document.body.classList.toggle("sidebar-open", open);
      button.setAttribute("aria-expanded", String(open));
      button.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
      if (buttonText) buttonText.textContent = open ? "Close" : "Menu";
    };

    const setDesktopCollapsed = (collapsed, persist = false) => {
      shell.classList.toggle("is-sidebar-collapsed", collapsed);
      button.setAttribute("aria-expanded", String(!collapsed));
      button.setAttribute("aria-label", collapsed ? "Expand navigation" : "Collapse navigation");
      if (persist) writeCollapsed(collapsed);
    };

    const syncMode = () => {
      setMobileOpen(false);
      if (mobile.matches) {
        shell.classList.remove("is-sidebar-collapsed");
      } else {
        setDesktopCollapsed(readCollapsed());
      }
    };

    button.addEventListener("click", () => {
      if (mobile.matches) {
        setMobileOpen(!shell.classList.contains("is-sidebar-open"));
      } else {
        setDesktopCollapsed(!shell.classList.contains("is-sidebar-collapsed"), true);
      }
    });

    backdrop.addEventListener("click", () => setMobileOpen(false));
    navigation.addEventListener("click", (event) => {
      if (mobile.matches && event.target.closest("a")) setMobileOpen(false);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && shell.classList.contains("is-sidebar-open")) {
        setMobileOpen(false);
        button.focus();
      }
    });
    mobile.addEventListener("change", syncMode);
    syncMode();
  });
})();
