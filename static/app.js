// AlphaFly Truco Uruguayo: Frontend Interactivo y Visualizador de Conectoma

let gameState = null;
let currentTab = "game-tab";

// Configuración de Palos
const SUIT_ICONS = {
  "Espada": "⚔️",
  "Basto": "🌿",
  "Oro": "🪙",
  "Copa": "🏆"
};

// Inicialización
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initEventListeners();
  fetchState();
  // Sondeo suave cada 2 segundos por si hay cambios
  setInterval(() => {
    if (currentTab === "game-tab") {
      fetchState(false);
    }
  }, 2500);
});

// Navegación por pestañas
function initTabs() {
  document.querySelectorAll(".nav-tab").forEach(tabBtn => {
    tabBtn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      
      tabBtn.classList.add("active");
      currentTab = tabBtn.dataset.tab;
      document.getElementById(currentTab).classList.add("active");

      // Si cambia a la pestaña del conectoma, redibujar canvas grande
      if (currentTab === "connectome-tab" && gameState) {
        renderLargeCanvas();
        renderSensoryInputs();
        renderMbonBars();
      }
    });
  });
}

// Event Listeners de Botones
function initEventListeners() {
  document.getElementById("btn-truco").addEventListener("click", () => callAction("/api/game/call_truco"));
  document.getElementById("btn-retruco").addEventListener("click", () => callAction("/api/game/call_retruco"));
  document.getElementById("btn-vale4").addEventListener("click", () => callAction("/api/game/call_vale4"));
  document.getElementById("btn-envido").addEventListener("click", () => callAction("/api/game/call_envido"));
  document.getElementById("btn-fold").addEventListener("click", () => callAction("/api/game/fold"));
  document.getElementById("btn-next-hand").addEventListener("click", () => callAction("/api/game/new_hand"));
  document.getElementById("btn-reset").addEventListener("click", () => {
    if (confirm("¿Deseas reiniciar la partida a 0-0?")) {
      callAction("/api/game/new_match");
    }
  });

  // Configuración de Chico (15 vs 30)
  const targetSelect = document.getElementById("target-score-select");
  if (targetSelect) {
    targetSelect.addEventListener("change", (e) => {
      const val = parseInt(e.target.value, 10);
      callAction("/api/game/config", { target_score: val });
    });
  }

  // Entrenamiento
  document.getElementById("btn-train-now").addEventListener("click", startTraining);
  document.getElementById("btn-reset-brain").addEventListener("click", resetBrain);
}

// Fetch State
async function fetchState(redrawCanvases = true) {
  try {
    const res = await fetch("/api/game/state");
    gameState = await res.json();
    renderUI(redrawCanvases);
  } catch (err) {
    console.error("Error al obtener estado:", err);
  }
}

// Llamadas API
async function callAction(endpoint, body = null) {
  try {
    const opts = { method: "POST" };
    if (body) {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(endpoint, opts);
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || "Acción no permitida");
      return;
    }
    gameState = await res.json();
    renderUI(true);
  } catch (err) {
    console.error(`Error en ${endpoint}:`, err);
  }
}

// Renderizado General
function renderUI(redrawCanvases = true) {
  if (!gameState) return;

  // 1. Tanteador
  document.getElementById("human-score").innerText = gameState.human_score;
  document.getElementById("fly-score").innerText = gameState.fly_score;
  const target = gameState.target_score || 15;
  document.getElementById("human-bar").style.width = `${Math.min(100, (gameState.human_score / target) * 100)}%`;
  document.getElementById("fly-bar").style.width = `${Math.min(100, (gameState.fly_score / target) * 100)}%`;

  const targetSelect = document.getElementById("target-score-select");
  if (targetSelect && document.activeElement !== targetSelect) {
    targetSelect.value = target.toString();
  }

  // 2. Muestra y Piezas
  const mCont = document.getElementById("muestra-container");
  mCont.innerHTML = "";
  if (gameState.muestra) {
    mCont.appendChild(createCardElement(gameState.muestra, gameState.muestra, false));
    const pz = gameState.muestra.piezas || [];
    document.getElementById("piezas-info").innerHTML = `
      Palo: <b>${gameState.muestra.suit}</b><br>
      Piezas: <b>${pz.join(", ")}</b>
    `;
  }

  // 3. Estado de la mano y apuestas
  document.getElementById("stake-badge").innerText = `Nivel: ${gameState.truco_level_name}`;
  const handRole = gameState.human_is_hand ? "Tú eres Mano" : "Mosca es Mano";
  document.getElementById("hand-meta-badge").innerText = `Mano #${gameState.hand_num} &bull; ${handRole}`;

  // Banner de turno y distintivo de tantos
  const turnBanner = document.getElementById("turn-banner");
  if (turnBanner) {
    turnBanner.innerText = gameState.turn_text || (gameState.can_human_play_card ? "🟢 Tu turno de jugar carta" : "🟡 Turno de la mosca...");
    turnBanner.className = "turn-banner " + (gameState.hand_over ? "hand-over" : (gameState.can_human_play_card ? "human-turn" : "fly-turn"));
  }

  const envBadge = document.getElementById("human-envido-badge");
  if (envBadge) {
    if (gameState.human_has_flor) {
      envBadge.innerText = `🌸 Flor: ${gameState.human_flor_points} tantos`;
      envBadge.style.borderColor = "#f59e0b";
      envBadge.style.color = "#fbbf24";
    } else {
      envBadge.innerText = `🔥 Envido: ${gameState.human_envido_points} tantos`;
      envBadge.style.borderColor = "rgba(255, 209, 92, 0.35)";
      envBadge.style.color = "var(--color-gold)";
    }
  }

  // 4. Cartas de la mosca (reverso)
  const flyHand = document.getElementById("fly-hand");
  flyHand.innerHTML = "";
  for (let i = 0; i < gameState.fly_card_count; i++) {
    const back = document.createElement("div");
    back.className = "card-back";
    flyHand.appendChild(back);
  }

  // 5. Cartas del Jugador Humano
  const humHand = document.getElementById("human-hand");
  humHand.innerHTML = "";
  if (!gameState.can_human_play_card) {
    humHand.classList.add("hand-disabled");
  } else {
    humHand.classList.remove("hand-disabled");
  }

  gameState.human_cards.forEach(c => {
    const cardEl = createCardElement(c, gameState.muestra, true);
    if (!gameState.can_human_play_card) {
      cardEl.classList.add("card-disabled");
    }
    cardEl.addEventListener("click", () => {
      if (gameState.pending_prompt) {
        alert("Debes responder al canto pendiente primero.");
        return;
      }
      if (!gameState.can_human_play_card) {
        alert("No es tu turno de jugar carta. Espera a que juegue la mosca.");
        return;
      }
      callAction("/api/game/play_card", { card_idx: c.idx });
    });
    humHand.appendChild(cardEl);
  });

  // 6. Bazas (Tapete de Juego)
  for (let i = 0; i < 3; i++) {
    const trickCardsEl = document.getElementById(`trick-cards-${i}`);
    trickCardsEl.innerHTML = "";
    const roundCards = gameState.table_cards[i] || [];
    
    roundCards.forEach(tc => {
      const cardEl = createCardElement(tc, gameState.muestra, false);
      cardEl.style.transform = tc.player === 0 ? "scale(0.85)" : "scale(0.85)";
      trickCardsEl.appendChild(cardEl);
    });

    const badge = document.getElementById(`trick-winner-0`.replace("0", i));
    badge.className = "trick-winner-badge";
    if (gameState.trick_history[i] !== undefined) {
      const w = gameState.trick_history[i];
      if (w === 0) {
        badge.innerText = "✓ Tú ganaste";
        badge.classList.add("winner-human");
      } else if (w === 1) {
        badge.innerText = "⚡ Mosca";
        badge.classList.add("winner-fly");
      } else {
        badge.innerText = "≈ Parda";
        badge.classList.add("winner-tie");
      }
    } else {
      badge.innerText = "";
    }
  }

  // 7. Botones de Acción
  document.getElementById("btn-envido").style.display = gameState.can_human_call_envido ? "inline-flex" : "none";
  document.getElementById("btn-truco").style.display = gameState.can_human_call_truco ? "inline-flex" : "none";
  document.getElementById("btn-retruco").style.display = gameState.can_human_call_retruco ? "inline-flex" : "none";
  document.getElementById("btn-vale4").style.display = gameState.can_human_call_vale4 ? "inline-flex" : "none";
  document.getElementById("btn-fold").style.display = (!gameState.hand_over && !gameState.match_over) ? "inline-flex" : "none";
  document.getElementById("btn-next-hand").style.display = (gameState.hand_over && !gameState.match_over) ? "inline-flex" : "none";

  // 8. Log de Jugadas
  const logBox = document.getElementById("game-log");
  logBox.innerHTML = "";
  (gameState.log || []).forEach(entry => {
    const d = document.createElement("div");
    d.className = `log-item log-${entry.category}`;
    d.innerHTML = `<span class="log-time">${entry.time}</span> <span class="log-text">${entry.text}</span>`;
    logBox.appendChild(d);
  });
  logBox.scrollTop = logBox.scrollHeight;

  // 9. Modal de Apuesta Pendiente
  const modal = document.getElementById("prompt-modal");
  if (gameState.pending_prompt) {
    modal.style.display = "flex";
    document.getElementById("modal-title").innerText = gameState.pending_prompt.title;

    // Vista previa de cartas y tantos dentro del modal
    const prevScore = document.getElementById("modal-preview-score");
    if (prevScore) {
      if (gameState.pending_prompt.type === "respond_envido") {
        if (gameState.human_has_flor) {
          prevScore.innerText = `🌸 ¡TIENES FLOR! (${gameState.human_flor_points} tantos)`;
        } else {
          prevScore.innerText = `Tus Tantos: ${gameState.human_envido_points}`;
        }
      } else {
        prevScore.innerText = `Apuesta actual: ${gameState.truco_level_name}`;
      }
    }

    const prevCards = document.getElementById("modal-preview-cards");
    if (prevCards) {
      prevCards.innerHTML = "";
      // 1. Mostrar la Muestra como referencia
      if (gameState.muestra) {
        const mWrap = document.createElement("div");
        mWrap.style.display = "flex";
        mWrap.style.flexDirection = "column";
        mWrap.style.alignItems = "center";
        mWrap.appendChild(createCardElement(gameState.muestra, gameState.muestra, false));
        const mTag = document.createElement("span");
        mTag.className = "modal-muestra-tag";
        mTag.innerText = "MUESTRA";
        mWrap.appendChild(mTag);
        prevCards.appendChild(mWrap);

        const sep = document.createElement("div");
        sep.className = "modal-muestra-separator";
        prevCards.appendChild(sep);
      }

      // 2. Mostrar las cartas del jugador humano
      gameState.human_cards.forEach(c => {
        const cEl = createCardElement(c, gameState.muestra, false);
        prevCards.appendChild(cEl);
      });
    }

    const actionsBox = document.getElementById("modal-actions");
    actionsBox.innerHTML = "";
    
    gameState.pending_prompt.options.forEach(opt => {
      const b = document.createElement("button");
      b.className = "modal-btn";
      if (opt.id === "quiero") b.classList.add("btn-accept-envite");
      else if (opt.id === "no_quiero") b.classList.add("btn-reject-envite");
      else b.classList.add("btn-raise-envite");
      b.innerText = opt.label;
      b.addEventListener("click", () => {
        modal.style.display = "none";
        if (gameState.pending_prompt.type === "respond_truco") {
          callAction("/api/game/respond_truco", { action: opt.id });
        } else if (gameState.pending_prompt.type === "respond_envido") {
          callAction("/api/game/respond_envido", { action: opt.id });
        }
      });
      actionsBox.appendChild(b);
    });
  } else {
    modal.style.display = "none";
  }

  // 10. Telemetría y Canvases
  if (gameState.telemetry && gameState.telemetry.drives) {
    const d = gameState.telemetry.drives;
    setMeter("drive-truco", "val-truco", d.truco);
    setMeter("drive-retruco", "val-retruco", d.retruco || d.vale4);
    setMeter("drive-accept", "val-accept", d.accept_truco);
    setMeter("drive-envido", "val-envido", d.envido);
    setMeter("drive-bluff", "val-bluff", d.bluff !== undefined ? d.bluff : 0.12);

    const stratBadge = document.getElementById("cognitive-strategy-badge");
    if (stratBadge && gameState.telemetry.cognitive_strategy) {
      stratBadge.innerText = gameState.telemetry.cognitive_text || "🟡 Cauto";
      stratBadge.className = "strategy-badge";
      if (gameState.telemetry.cognitive_strategy === "FAROL") {
        stratBadge.classList.add("strategy-bluff");
      } else if (gameState.telemetry.cognitive_strategy === "VALOR") {
        stratBadge.classList.add("strategy-valor");
      } else {
        stratBadge.classList.add("strategy-defense");
      }
    }

    const dop = gameState.telemetry.last_dopamine || 0;
    if (dop > 0) {
      document.getElementById("dop-pam").innerText = `🍬 PAM (Premio): +${dop.toFixed(1)}`;
      document.getElementById("dop-pam").style.boxShadow = "0 0 10px rgba(0,240,168,0.5)";
    } else if (dop < 0) {
      document.getElementById("dop-ppl1").innerText = `⚡ PPL1 (Aversión): ${dop.toFixed(1)}`;
      document.getElementById("dop-ppl1").style.boxShadow = "0 0 10px rgba(255,71,114,0.5)";
    }
    document.getElementById("sparse-badge").innerText = `${gameState.telemetry.sparse_pct}% KC Activas (${gameState.telemetry.active_kc_count}/500)`;
  }

  if (redrawCanvases) {
    renderMiniCanvas();
    if (currentTab === "connectome-tab") {
      renderLargeCanvas();
      renderSensoryInputs();
      renderMbonBars();
    }
  }
}

function setMeter(barId, valId, prob) {
  const pct = Math.round((prob || 0) * 100);
  const bar = document.getElementById(barId);
  const val = document.getElementById(valId);
  if (bar) bar.style.width = `${pct}%`;
  if (val) val.innerText = `${pct}%`;
}

// Renderizado de Carta Estilo Baraja Española
function createCardElement(card, muestra, isInteractive = false) {
  const el = document.createElement("div");
  el.className = `truco-card suit-${card.suit}`;
  
  const icon = SUIT_ICONS[card.suit] || "🃏";
  const piezas = muestra ? (muestra.piezas || []) : [];
  const isPieza = (muestra && card.suit === muestra.suit && piezas.includes(card.number));
  const isMacho = (card.number === 1 && card.suit === "Espada");
  const isHembra = (card.number === 1 && card.suit === "Basto");
  const is7Bravo = (card.number === 7 && (card.suit === "Espada" || card.suit === "Oro"));

  if (isPieza) el.classList.add("is-pieza");
  else if (isMacho || isHembra || is7Bravo) el.classList.add("is-mata");

  let badgeHtml = "";
  if (isPieza) {
    const pIdx = piezas.indexOf(card.number);
    const pNames = ["2 (MÁXIMA)", "4", "5", "11", "10"];
    badgeHtml = `<div class="card-badge badge-pieza">★ PIEZA ${pNames[pIdx] || ""}</div>`;
  } else if (isMacho) {
    badgeHtml = `<div class="card-badge badge-mata">⚔ MACHO</div>`;
  } else if (isHembra) {
    badgeHtml = `<div class="card-badge badge-mata">🌿 HEMBRA</div>`;
  } else if (is7Bravo) {
    badgeHtml = `<div class="card-badge badge-mata">⚡ 7 BRAVO</div>`;
  }

  el.innerHTML = `
    <div class="card-top">
      <span class="card-num">${card.number}</span>
      <span class="card-suit-icon">${icon}</span>
    </div>
    <div class="card-center">
      <span>${icon}</span>
    </div>
    ${badgeHtml}
  `;
  return el;
}

// Canvas Mini (Pestaña de Juego)
function renderMiniCanvas() {
  const canvas = document.getElementById("mini-kc-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const coords = gameState.kc_coords || [];
  const acts = (gameState.telemetry && gameState.telemetry.kc_activity) ? gameState.telemetry.kc_activity : [];

  // Dibujar neuronas
  for (let i = 0; i < coords.length; i++) {
    const c = coords[i];
    const act = acts[i] || 0;
    const x = c.x * w;
    const y = c.y * h;

    if (act > 0) {
      // Neurona activa (resplandor bioluminiscente)
      ctx.beginPath();
      ctx.arc(x, y, 3.5 + act * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 168, ${0.4 + act * 0.6})`;
      ctx.shadowColor = "#00f0a8";
      ctx.shadowBlur = 8;
      ctx.fill();
    } else {
      // Neurona inactiva
      ctx.beginPath();
      ctx.arc(x, y, 1.8, 0, Math.PI * 2);
      ctx.fillStyle = "#1e293b";
      ctx.shadowBlur = 0;
      ctx.fill();
    }
  }
}

// Canvas Grande (Pestaña de Conectoma)
function renderLargeCanvas() {
  const canvas = document.getElementById("large-kc-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const coords = gameState.kc_coords || [];
  const acts = (gameState.telemetry && gameState.telemetry.kc_activity) ? gameState.telemetry.kc_activity : [];

  // 1. Dibujar dendritas de fondo
  ctx.strokeStyle = "rgba(45, 65, 95, 0.25)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i < coords.length; i += 6) {
    if (i + 1 < coords.length) {
      ctx.beginPath();
      ctx.moveTo(coords[i].x * w, coords[i].y * h);
      ctx.lineTo(coords[i+1].x * w, coords[i+1].y * h);
      ctx.stroke();
    }
  }

  // 2. Dibujar enlaces de neuronas activas
  const activeIndices = [];
  for (let i = 0; i < acts.length; i++) {
    if (acts[i] > 0) activeIndices.push(i);
  }

  ctx.strokeStyle = "rgba(0, 240, 168, 0.35)";
  ctx.lineWidth = 1;
  for (let k = 0; k < activeIndices.length; k += 2) {
    if (k + 1 < activeIndices.length) {
      const i1 = activeIndices[k];
      const i2 = activeIndices[k+1];
      ctx.beginPath();
      ctx.moveTo(coords[i1].x * w, coords[i1].y * h);
      ctx.lineTo(coords[i2].x * w, coords[i2].y * h);
      ctx.stroke();
    }
  }

  // 3. Dibujar nodos de las 500 KCs
  for (let i = 0; i < coords.length; i++) {
    const c = coords[i];
    const act = acts[i] || 0;
    const x = c.x * w;
    const y = c.y * h;

    if (act > 0) {
      ctx.beginPath();
      ctx.arc(x, y, 4.5 + act * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 168, ${0.4 + act * 0.6})`;
      ctx.shadowColor = "#00f0a8";
      ctx.shadowBlur = 12;
      ctx.fill();

      // Centro blanco brillante
      ctx.beginPath();
      ctx.arc(x, y, 1.8, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.shadowBlur = 0;
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "#1c2638";
      ctx.shadowBlur = 0;
      ctx.fill();
    }
  }

  // Configurar Tooltip
  setupLargeCanvasTooltip(canvas, coords, acts);
}

function setupLargeCanvasTooltip(canvas, coords, acts) {
  const tooltip = document.getElementById("kc-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / canvas.width;
    const my = (e.clientY - rect.top) / canvas.height;

    let found = null;
    for (let i = 0; i < coords.length; i++) {
      const dx = coords[i].x - mx;
      const dy = coords[i].y - my;
      if (dx * dx + dy * dy < 0.0006) {
        found = { i, c: coords[i], act: acts[i] || 0 };
        break;
      }
    }

    if (found) {
      tooltip.style.display = "block";
      tooltip.style.left = `${e.clientX - rect.left + 14}px`;
      tooltip.style.top = `${e.clientY - rect.top - 10}px`;
      tooltip.innerHTML = `
        <b>Kenyon Cell #${found.i + 1}</b><br>
        BodyID: <code>${found.c.id}</code><br>
        Región: <b>${found.c.region}</b><br>
        Disparo: <b>${(found.act * 100).toFixed(1)}%</b>
      `;
    } else {
      tooltip.style.display = "none";
    }
  };
  canvas.onmouseleave = () => { tooltip.style.display = "none"; };
}

// Entradas Sensoriales PN
function renderSensoryInputs() {
  const box = document.getElementById("inputs-list");
  if (!box || !gameState.telemetry || !gameState.telemetry.inputs) return;
  const inps = gameState.telemetry.inputs;
  const labels = [
    "Poder Carta 1 en mano", "Poder Carta 2 en mano", "Poder Carta 3 en mano",
    "Puntos de Envido Propios", "¿La Muestra es Pieza?",
    "Poder Mesa Baza 1", "Poder Mesa Baza 2", "Poder Mesa Baza 3", "Poder Mesa Extra",
    "Baza 1 Activa", "Baza 2 Activa", "Baza 3 Activa",
    "Nivel de Apuesta Truco", "Envido Ya Cantado", "Cartas Restantes", "Bias Biológico"
  ];

  box.innerHTML = "";
  labels.forEach((lbl, idx) => {
    const val = inps[idx] !== undefined ? inps[idx] : 0.0;
    const row = document.createElement("div");
    row.className = "input-row";
    row.innerHTML = `
      <span>${lbl}:</span>
      <b>${val.toFixed(2)}</b>
    `;
    box.appendChild(row);
  });
}

// Barras MBON
function renderMbonBars() {
  const box = document.getElementById("mbon-bars-container");
  if (!box || !gameState.telemetry || !gameState.telemetry.mbon_activity) return;
  const mb = gameState.telemetry.mbon_activity;
  
  const mbonRoles = {
    0: "Carta #1 (Jugar)", 1: "Carta #2 (Jugar)", 2: "Carta #3 (Jugar)",
    3: "Truco (Cantar)", 4: "Truco (Silencio)",
    5: "Truco (Quiero)", 6: "Truco (No Quiero)",
    7: "Envido (Cantar)", 8: "Envido (Silencio)",
    9: "Re-truco (Subir)", 10: "Re-truco (Freno)",
    11: "Vale 4 (Subir)", 12: "Vale 4 (Freno)",
    13: "Envido (Quiero)", 14: "Envido (No Quiero)"
  };

  box.innerHTML = "";
  mb.forEach((val, idx) => {
    const name = mbonRoles[idx] || `MBON #${idx + 1}`;
    const row = document.createElement("div");
    row.className = "mbon-row";
    const barW = Math.min(100, Math.max(0, val * 50));
    row.innerHTML = `
      <span style="width: 130px; color: #94a3b8;">${name}:</span>
      <div class="drive-meter" style="flex:1;"><div class="drive-bar" style="width: ${barW}%"></div></div>
      <span style="width: 38px; text-align:right;">${val.toFixed(2)}</span>
    `;
    box.appendChild(row);
  });
}

// Gimnasio de Entrenamiento
async function startTraining() {
  const ep = parseInt(document.getElementById("train-episodes").value);
  const lr = parseFloat(document.getElementById("train-lr").value);

  const statusBox = document.getElementById("train-status");
  const resultsBox = document.getElementById("train-results");
  const btn = document.getElementById("btn-train-now");

  statusBox.style.display = "flex";
  resultsBox.style.display = "none";
  btn.disabled = true;

  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes: ep, lr: lr })
    });
    const data = await res.json();

    document.getElementById("res-episodes").innerText = data.episodes;
    document.getElementById("res-winrate").innerText = `${data.final_win_rate}%`;
    document.getElementById("res-flypts").innerText = data.fly_total_pts;
    document.getElementById("res-dopamine").innerText = data.total_dopamine;

    resultsBox.style.display = "block";
    fetchState(true);
  } catch (err) {
    alert("Error durante el entrenamiento: " + err);
  } finally {
    statusBox.style.display = "none";
    btn.disabled = false;
  }
}

async function resetBrain() {
  if (!confirm("¿Deseas restaurar la mosca al conectoma biológico vírgen sin entrenar?")) return;
  try {
    await fetch("/api/brain/reset", { method: "POST" });
    alert("Pesos biológicos restaurados.");
    fetchState(true);
  } catch (err) {
    alert("Error al restaurar: " + err);
  }
}
