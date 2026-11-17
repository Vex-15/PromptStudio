/* Shared theme + motion controls, used by the studio page and the docs page.
   Both pages carry #theme-toggle / #motion-toggle buttons; app.js listens for
   the "themechange" and "motionchange" events for its own theme-dependent bits. */

const THEME_KEY = "prompt-studio-theme";
const MOTION_KEY = "prompt-studio-motion";

function motionOn() {
  return document.documentElement.dataset.motion !== "off";
}

function applyTheme(theme, persist) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === "dark";
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.textContent = dark ? "Light" : "Dark";
    btn.setAttribute("aria-pressed", String(dark));
    btn.title = dark ? "Switch to light theme" : "Switch to dark theme";
  }
  if (persist) {
    try { localStorage.setItem(THEME_KEY, theme); } catch { /* storage blocked */ }
  }
  document.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch { /* storage blocked */ }
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  applyTheme(saved || (media.matches ? "dark" : "light"), false);

  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next, true);
    });
  }

  media.addEventListener?.("change", (event) => {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch { /* storage blocked */ }
    if (!stored) applyTheme(event.matches ? "dark" : "light", false);
  });
}

function applyMotion(on, persist) {
  document.documentElement.dataset.motion = on ? "on" : "off";
  const btn = document.getElementById("motion-toggle");
  if (btn) {
    btn.textContent = on ? "Motion on" : "Motion off";
    btn.setAttribute("aria-pressed", String(on));
    btn.title = on ? "Turn animations off" : "Turn animations on";
  }
  if (persist) {
    try { localStorage.setItem(MOTION_KEY, on ? "on" : "off"); } catch { /* storage blocked */ }
  }
  document.dispatchEvent(new CustomEvent("motionchange", { detail: { on } }));
}

function initMotion() {
  let saved = null;
  try { saved = localStorage.getItem(MOTION_KEY); } catch { /* storage blocked */ }
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  applyMotion(saved ? saved === "on" : !reduced, false);

  const btn = document.getElementById("motion-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      applyMotion(document.documentElement.dataset.motion === "off", true);
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initMotion();
});
