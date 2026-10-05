/* AI Companion custom UI — vanilla JS, no framework. */
(() => {
  const $ = (s) => document.querySelector(s);
  const stream = $("#stream"), emptyState = $("#emptyState");
  const chatSelect = $("#chatSelect"), newChatName = $("#newChatName");
  const chatInput = $("#chatInput"), statusPill = $("#statusPill");
  const backendSelect = $("#backendSelect"), modelSelect = $("#modelSelect");
  const claudeModel = $("#claudeModel"), gpuLayers = $("#gpuLayers");
  const traitsBox = $("#traitsBox");

  const state = {
    chat_id: null, backend: "Local (GGUF)", model: "",
    claude: "", traits: new Set(), busy: false,
  };

  /* ── theme (vibe × palette), persisted ── */
  const root = document.documentElement;
  const CUSTOM_DEFAULTS = { bg: "#fff7f3", card: "#fffdfc", ink: "#38202e", accent: "#d63d7c", accent2: "#e8814a" };
  const CUSTOM_KEYS = Object.keys(CUSTOM_DEFAULTS);
  const CUSTOM_INPUTS = { bg: "#cBg", card: "#cCard", ink: "#cInk", accent: "#cAccent", accent2: "#cAccent2" };

  function customColors() {
    try {
      return { ...CUSTOM_DEFAULTS, ...JSON.parse(localStorage.getItem("ac.custom") || "{}") };
    } catch {
      return { ...CUSTOM_DEFAULTS };
    }
  }
  function luminance(hex) {
    const c = hex.replace("#", "");
    const f = (i) => {
      const x = parseInt(c.substr(i, 2), 16) / 255;
      return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(0) + 0.7152 * f(2) + 0.0722 * f(4);
  }
  function applyCustomColors() {
    const c = customColors();
    for (const el of [root, document.body]) {
      for (const k of CUSTOM_KEYS) el.style.setProperty("--" + k, c[k]);
      el.style.setProperty("--user-ink", luminance(c.accent) > 0.4 ? "#1c1a19" : "#ffffff");
      el.style.setProperty("--glow", c.accent + "47");
    }
  }
  function clearCustomVars() {
    for (const el of [root, document.body]) {
      for (const k of [...CUSTOM_KEYS, "user-ink", "glow"]) el.style.removeProperty("--" + k);
    }
  }
  function syncCustomInputs() {
    const c = customColors();
    for (const [k, sel] of Object.entries(CUSTOM_INPUTS)) {
      const input = $(sel);
      if (input) input.value = c[k];
    }
  }
  function applyTheme() {
    const vibe = localStorage.getItem("ac.vibe") || "cozy";
    const pal = localStorage.getItem("ac.palette") || "rose";
    root.dataset.vibe = vibe; root.dataset.palette = pal;
    document.body.dataset.vibe = vibe; document.body.dataset.palette = pal;
    if (pal === "custom") applyCustomColors(); else clearCustomVars();
    const editor = $("#customEditor");
    if (editor) editor.hidden = pal !== "custom";
    document.querySelectorAll("[data-set-vibe]").forEach(b =>
      b.classList.toggle("active", b.dataset.setVibe === vibe));
    document.querySelectorAll("[data-set-palette]").forEach(b =>
      b.classList.toggle("active", b.dataset.setPalette === pal));
  }
  document.querySelectorAll("[data-set-vibe]").forEach(b =>
    b.addEventListener("click", () => { localStorage.setItem("ac.vibe", b.dataset.setVibe); applyTheme(); }));
  document.querySelectorAll("[data-set-palette]").forEach(b =>
    b.addEventListener("click", () => { localStorage.setItem("ac.palette", b.dataset.setPalette); applyTheme(); }));
  for (const [k, sel] of Object.entries(CUSTOM_INPUTS)) {
    const input = $(sel);
    if (input) input.addEventListener("input", () => {
      const c = { ...customColors(), [k]: input.value };
      localStorage.setItem("ac.custom", JSON.stringify(c));
      if ((localStorage.getItem("ac.palette") || "rose") === "custom") applyCustomColors();
    });
  }
  syncCustomInputs();
  applyTheme();

  /* ── panels / modal ── */
  const scrim = $("#scrim");
  const panels = [$("#personaPanel"), $("#settingsPanel")];
  const modal = $("#stylesModal");
  function refreshChrome() {
    const anyOpen = panels.some(p => p.classList.contains("open")) || modal.classList.contains("open");
    scrim.hidden = !anyOpen;
    [...panels, modal].forEach(el => el.setAttribute("aria-hidden", String(!el.classList.contains("open"))));
  }
  function closeAll() { panels.forEach(p => p.classList.remove("open")); modal.classList.remove("open"); refreshChrome(); }
  $("#personaBtn").onclick = () => { closeAll(); $("#personaPanel").classList.add("open"); refreshChrome(); };
  $("#settingsBtn").onclick = () => { closeAll(); $("#settingsPanel").classList.add("open"); refreshChrome(); };
  $("#stylesBtn").onclick = () => { closeAll(); modal.classList.add("open"); refreshChrome(); };
  document.querySelectorAll("[data-close]").forEach(b => (b.onclick = closeAll));
  scrim.onclick = closeAll;
  addEventListener("keydown", e => { if (e.key === "Escape") closeAll(); });

  function toast(msg) {
    const t = document.createElement("div");
    t.className = "toast"; t.textContent = msg;
    $("#toasts").appendChild(t);
    setTimeout(() => t.remove(), 3200);
  }

  const api = async (path, opts = {}) => {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    return res.json();
  };

  /* ── rendering ── */
  function initials(name, fallback) {
    const n = (name || "").trim();
    return n ? n[0].toUpperCase() : fallback;
  }
  function companionName() { return $("#companionName").value.trim() || "Companion"; }

  function addMsg(role, content, opts = {}) {
    emptyState.style.display = "none";
    const wrap = document.createElement("div");
    wrap.className = `msg ${role === "user" ? "user" : "ai"}`;
    wrap.style.setProperty("--i", String(stream.childElementCount % 6));
    const av = document.createElement("div");
    av.className = "avatar";
    av.textContent = role === "user"
      ? initials($("#userName").value, "Y")
      : initials(companionName(), "✦");
    const bub = document.createElement("div");
    bub.className = "bubble";
    const who = document.createElement("div");
    who.className = "who";
    const name = document.createElement("span");
    name.textContent = role === "user" ? ($("#userName").value.trim() || "You") : companionName();
    who.appendChild(name);
    if (opts.tag) {
      const t = document.createElement("time");
      t.textContent = opts.tag;
      who.appendChild(t);
    }
    const txt = document.createElement("div");
    txt.textContent = content;
    bub.append(who, txt);
    wrap.append(av, bub);
    stream.appendChild(wrap);
    wrap.scrollIntoView({ behavior: "smooth", block: "end" });
    return wrap;
  }

  function now() {
    const d = new Date();
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
  }

  function setThinking(on) {
    document.body.dataset.state = on ? "thinking" : "idle";
    statusPill.textContent = on ? "thinking…" : statusPill.textContent;
    setTyping(on);
  }

  function setTyping(on) {
    let el = $("#typingRow");
    if (on && !el) {
      el = document.createElement("div");
      el.className = "msg ai"; el.id = "typingRow";
      el.innerHTML = `<div class="avatar">✦</div><div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>`;
      stream.appendChild(el);
      el.scrollIntoView({ behavior: "smooth", block: "end" });
    } else if (!on && el) el.remove();
  }

  async function loadHistory(mode = "recent") {
    if (!state.chat_id) return;
    const { history, status } = await api(`/api/history?chat_id=${encodeURIComponent(state.chat_id)}&mode=${mode}`);
    stream.innerHTML = "";
    emptyState.style.display = history.length ? "none" : "";
    history.forEach((m, i) => addMsg(m.role === "user" ? "user" : "ai", m.content, { tag: "#" + String(Math.floor(i / 2) + 1).padStart(3, "0") }));
    statusPill.textContent = status || "";
  }

  /* ── boot ── */
  async function boot() {
    const cfg = await api("/api/config");
    backendSelect.innerHTML = cfg.backends.map(b => `<option>${b}</option>`).join("");
    state.backend = localStorage.getItem("ac.backend") || cfg.backends[0];
    backendSelect.value = state.backend;
    claudeModel.value = localStorage.getItem("ac.claude") || cfg.default_claude_model;
    $("#keyNote").textContent = cfg.has_openrouter_key
      ? "OpenRouter key detected on server. ✓"
      : "No OPENROUTER_API_KEY on server — Claude backend will refuse until you add it to .env and restart server.py.";

    const { models } = await api("/api/models");
    modelSelect.innerHTML = models.length ? models.map(m => `<option>${m}</option>`).join("") : `<option value="">(no .gguf in models/)</option>`;
    state.model = localStorage.getItem("ac.model") || models[0] || "";
    if (state.model) modelSelect.value = state.model;
    syncBackendUI();

    const { chats } = await api("/api/chats");
    chatSelect.innerHTML = chats.map(c => `<option>${c}</option>`).join("");
    state.chat_id = localStorage.getItem("ac.chat") || chats[0];
    chatSelect.value = state.chat_id;

    traitsBox.innerHTML = (cfg.personalities || []).map(t =>
      `<button type="button" class="chip" aria-pressed="false">${t}</button>`).join("");
    traitsBox.querySelectorAll(".chip").forEach(ch =>
      ch.onclick = () => {
        const on = ch.getAttribute("aria-pressed") === "true";
        ch.setAttribute("aria-pressed", String(!on));
        !on ? state.traits.add(ch.textContent) : state.traits.delete(ch.textContent);
      });

    await loadPersona();
    await loadHistory();
    updateBrand();
  }

  function syncBackendUI() {
    const isClaude = state.backend === "Claude (OpenRouter)";
    $("#modelRow").style.display = isClaude ? "none" : "";
    $("#claudeRow").style.display = isClaude ? "" : "none";
    updateBrand();
  }
  function updateBrand() {
    const initial = initials(companionName(), "A");
    const mark = $("#brandMark");
    mark.childNodes[0].textContent = initial;
    $("#brandName").textContent = companionName();
    const orbGlyph = $("#emptyAvatar span") || $("#emptyAvatar");
    orbGlyph.textContent = initials(companionName(), "✦");
    $("#backendPill").textContent =
      state.backend === "Claude (OpenRouter)" ? "claude · openrouter" : `local · ${state.model || "no model yet"}`;
  }

  async function loadPersona() {
    if (!state.chat_id) return;
    const { persona } = await api(`/api/persona?chat_id=${encodeURIComponent(state.chat_id)}`);
    $("#userName").value = persona.user_name || "";
    $("#companionName").value = persona.companion_name || "";
    $("#userGender").value = persona.user_gender || "";
    $("#companionGender").value = persona.companion_gender || "";
    $("#customPersona").value = persona.custom_personality || "";
    state.traits = new Set(persona.personality_traits || []);
    traitsBox.querySelectorAll(".chip").forEach(ch =>
      ch.setAttribute("aria-pressed", String(state.traits.has(ch.textContent))));
    updateBrand();
  }

  /* ── events ── */
  chatSelect.onchange = async () => {
    state.chat_id = chatSelect.value;
    localStorage.setItem("ac.chat", state.chat_id);
    await loadPersona(); await loadHistory(); updateBrand();
  };
  $("#createChatBtn").onclick = async () => {
    const name = newChatName.value.trim();
    if (!name) return toast("Type a chat name first.");
    const r = await api("/api/chats", { method: "POST", body: JSON.stringify({ name, current: state.chat_id }) });
    chatSelect.innerHTML = r.chats.map(c => `<option>${c}</option>`).join("");
    state.chat_id = r.selected; chatSelect.value = state.chat_id;
    localStorage.setItem("ac.chat", state.chat_id);
    newChatName.value = "";
    await loadPersona(); await loadHistory(); updateBrand();
    toast(r.message);
  };
  backendSelect.onchange = () => { state.backend = backendSelect.value; localStorage.setItem("ac.backend", state.backend); syncBackendUI(); };
  modelSelect.onchange = () => { state.model = modelSelect.value; localStorage.setItem("ac.model", state.model); updateBrand(); };
  claudeModel.onchange = () => localStorage.setItem("ac.claude", claudeModel.value.trim());

  $("#savePersonaBtn").onclick = async () => {
    const body = {
      chat_id: state.chat_id,
      user_name: $("#userName").value, companion_name: $("#companionName").value,
      user_gender: $("#userGender").value, companion_gender: $("#companionGender").value,
      traits: [...state.traits], custom_personality: $("#customPersona").value,
    };
    const r = await api("/api/persona", { method: "POST", body: JSON.stringify(body) });
    $("#personaMsg").textContent = r.message;
    updateBrand(); toast(r.message);
  };

  $("#loadAllBtn").onclick = () => loadHistory("all").catch(e => toast(e.message));
  $("#clearBtn").onclick = async () => {
    if (!confirm(`Delete all memory for "${state.chat_id}"?`)) return;
    const r = await api(`/api/history?chat_id=${encodeURIComponent(state.chat_id)}`, { method: "DELETE" });
    stream.innerHTML = ""; emptyState.style.display = "";
    statusPill.textContent = r.status; toast(r.status);
  };

  document.querySelectorAll(".hint").forEach(h =>
    h.onclick = () => { chatInput.value = h.textContent.replace(/[“”]/g, ""); chatInput.focus(); autoGrow(); });

  function autoGrow() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
  }
  chatInput.addEventListener("input", autoGrow);

  $("#composer").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text || state.busy) return;
    state.busy = true;
    addMsg("user", text, { tag: now() });
    chatInput.value = ""; autoGrow();
    setThinking(true);
    try {
      const r = await api("/api/chat", { method: "POST", body: JSON.stringify({
        message: text, chat_id: state.chat_id,
        model_name: state.model, gpu_layers: parseInt(gpuLayers.value || "0", 10),
        backend: state.backend, claude_model: claudeModel.value.trim(),
      })});
      setThinking(false);
      if (!r.handled) return toast(r.status || "Nothing sent.");
      const reply = r.exchange.filter(m => m.role !== "user").pop();
      if (reply) addMsg("ai", reply.content, { tag: now() });
      statusPill.textContent = r.status || "";
    } catch (err) {
      setThinking(false);
      toast(err.message);
    } finally { state.busy = false; }
  });

  boot().catch(e => { statusPill.textContent = "Server error: " + e.message; });
})();
