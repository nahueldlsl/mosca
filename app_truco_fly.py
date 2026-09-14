"""
Servidor Web FastAPI para el Duelo y Panel Interactivo de Truco Uruguayo:
Humano vs Cerebro de la Mosca (FlyEM Male CNS v1.0).
Incluye telemetría en tiempo real del conectoma de 500 KCs y MBONs.
"""

import os
import random
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from truco_engine import (
    Card, create_deck, card_power, calculate_envido,
    get_effective_piezas, has_flor, calculate_flor_points,
    card_label, TRUCO_VALUES, TRUCO_NAMES, TRUCO_REJECT_POINTS
)
from fly_brain_agent import FlyBrainTrucoAgent
from train_fly_truco import HeuristicBot, simulate_hand, train_fly_session

app = FastAPI(title="AlphaFly Truco Uruguayo")

# Montar archivos estáticos
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DATA_DIR = Path(__file__).parent / "data"

# Estado global de la partida
class GameSession:
    def __init__(self):
        self.fly = FlyBrainTrucoAgent()
        model_path = "data/fly_truco_model.npz"
        if os.path.exists(model_path):
            self.fly.load_model(model_path)

        self.human_score = 0
        self.fly_score = 0
        self.target_score = 15
        self.hand_num = 0

        # Estado de la mano activa
        self.muestra: Optional[Card] = None
        self.human_cards: List[Card] = []
        self.fly_cards: List[Card] = []
        self.table_cards: List[List[Dict]] = [[], [], []] # Bazas 0, 1, 2
        self.trick_idx = 0
        self.trick_history: List[int] = []
        self.leader = 0 # 0: Humano, 1: Mosca
        self.human_is_hand = True
        self.hand_over = False
        self.match_over = False

        # Apuestas
        self.envido_state = "NONE" # NONE, CALLED, RESOLVED, CANCELED
        self.envido_caller = None
        self.truco_level = 0 # 0: none, 1: truco, 2: retruco, 3: vale 4
        self.canto_holder: Optional[int] = None # 0: humano, 1: mosca

        # Turno actual (0: Humano, 1: Mosca)
        self.current_turn = 0

        # Esperas de respuesta del humano
        self.pending_prompt: Optional[Dict] = None

        self.log: List[Dict] = []
        self.latest_telemetry: Dict = {}

        # Generar posiciones 2D anatómicas estables para las 500 KCs
        self.kc_coords = self._generate_kc_morphology_coords(500)

        # Iniciar primera mano
        self.start_new_hand()

    def _generate_kc_morphology_coords(self, n: int) -> List[Dict]:
        """Genera coordenadas 2D fieles a la morfología del Mushroom Body (Calyx y Lóbulos)."""
        rng = np.random.RandomState(42)
        coords = []
        for i in range(n):
            if i < 200:
                # Calyx (Cáliz bulboso superior donde llegan los PNs)
                r = rng.uniform(0.05, 0.45)
                theta = rng.uniform(0, 2 * np.pi)
                cx, cy = 0.5, 0.25
                x = cx + r * np.cos(theta) * 0.7
                y = cy + r * np.sin(theta) * 0.45
                region = "Calyx"
            elif i < 300:
                # Pedúnculo (Haz axonal central)
                t = rng.uniform(0.0, 1.0)
                x = 0.5 + rng.normal(0, 0.03)
                y = 0.45 + t * 0.22
                region = "Pedunculus"
            elif i < 400:
                # Lóbulo Vertical (Alfa / Alfa')
                t = rng.uniform(0.0, 1.0)
                x = 0.48 - t * 0.28 + rng.normal(0, 0.035)
                y = 0.67 + t * 0.25 + rng.normal(0, 0.035)
                region = "Lobe_Alpha"
            else:
                # Lóbulo Medial (Beta / Gamma)
                t = rng.uniform(0.0, 1.0)
                x = 0.52 + t * 0.38 + rng.normal(0, 0.035)
                y = 0.68 + t * 0.08 + rng.normal(0, 0.035)
                region = "Lobe_BetaGamma"

            coords.append({
                "id": int(self.fly.kc_ids[i]) if i < len(self.fly.kc_ids) else i,
                "x": round(float(np.clip(x, 0.05, 0.95)), 4),
                "y": round(float(np.clip(y, 0.05, 0.95)), 4),
                "region": region
            })
        return coords

    def add_log(self, text: str, category: str = "info"):
        self.log.append({
            "text": text,
            "category": category,
            "time": time.strftime("%H:%M:%S")
        })
        if len(self.log) > 60:
            self.log.pop(0)

    def start_new_hand(self):
        if self.human_score >= self.target_score or self.fly_score >= self.target_score:
            self.match_over = True
            return

        self.hand_num += 1
        deck = create_deck()
        random.shuffle(deck)
        self.muestra = deck.pop()
        self.human_cards = [deck.pop() for _ in range(3)]
        self.fly_cards = [deck.pop() for _ in range(3)]

        self.human_is_hand = (self.hand_num % 2 == 1)
        self.leader = 0 if self.human_is_hand else 1
        self.current_turn = self.leader
        self.trick_idx = 0
        self.table_cards = [[], [], []]
        self.trick_history = []
        self.hand_over = False
        self.truco_level = 0
        self.canto_holder = None
        self.pending_prompt = None

        self.add_log(f"--- MANO #{self.hand_num} --- Muestra: {card_label(self.muestra, self.muestra)}", "header")

        # 1. FLOR (Reglamento Oficial: 2 o 3 piezas, 1 pieza + 2 del palo, o 3 del mismo palo)
        h_flor = has_flor(self.human_cards, self.muestra)
        f_flor = has_flor(self.fly_cards, self.muestra)

        if h_flor and f_flor:
            hp = calculate_flor_points(self.human_cards, self.muestra)
            fp = calculate_flor_points(self.fly_cards, self.muestra)
            self.add_log(f"🌸 ¡Doble Flor! Tú: {hp} tantos | Mosca: {fp} tantos.", "flor")
            if hp > fp:
                self.human_score += 3
                self.add_log(">> ¡Ganaste el lance de Flor! (+3 pts)", "win")
                self.fly.apply_dopamine_reward(-1.0)
            elif fp > hp:
                self.fly_score += 3
                self.add_log(">> La mosca gana la Flor (+3 pts)", "loss")
                self.fly.apply_dopamine_reward(1.5)
            else:
                if self.human_is_hand:
                    self.human_score += 3
                    self.add_log(">> Empate en Flor: ¡Ganas por ser mano! (+3 pts)", "win")
                    self.fly.apply_dopamine_reward(-1.0)
                else:
                    self.fly_score += 3
                    self.add_log(">> Empate en Flor: Mosca gana por ser mano (+3 pts)", "loss")
                    self.fly.apply_dopamine_reward(1.5)
            self.envido_state = "CANCELED"
        elif h_flor:
            hp = calculate_flor_points(self.human_cards, self.muestra)
            self.human_score += 3
            self.add_log(f"🌸 ¡¡TIENES FLOR!! ({hp} tantos) -> ¡Sumas 3 puntos!", "flor")
            self.envido_state = "CANCELED"
        elif f_flor:
            fp = calculate_flor_points(self.fly_cards, self.muestra)
            self.fly_score += 3
            self.add_log(f"🌸 ¡La mosca zumba y canta FLOR! ({fp} tantos) -> Suma 3 puntos.", "flor")
            self.fly.apply_dopamine_reward(1.5)
            self.envido_state = "CANCELED"
        else:
            self.envido_state = "NONE"

        # Actualizar telemetría inicial
        st = self.fly.encode_state(self.fly_cards, self.muestra, [], 0, 0, False)
        self.latest_telemetry = self.fly.get_detailed_telemetry(st)

        # Si no hay flor y lidera la mosca: evalúa si cantar envido
        if self.envido_state == "NONE" and not self.human_is_hand:
            if self.fly.decide_envido(st):
                self.envido_state = "CALLED"
                self.envido_caller = 1
                self.add_log("🐝 ¡La mosca te mira fijo y canta: ¡¡ENVIDO!!", "truco")
                self.pending_prompt = {
                    "type": "respond_envido",
                    "title": "La mosca cantó ENVIDO",
                    "options": [
                        {"id": "quiero", "label": "Quiero (disputar tantos)"},
                        {"id": "no_quiero", "label": "No quiero (ceder 1 punto)"}
                    ]
                }
                return

        # Si lidera la mosca y no hay prompt pendiente, abre la primera baza
        if not self.human_is_hand and not self.pending_prompt:
            self._fly_turn()

    def _fly_play_card_only(self):
        """Ejecuta única y exclusivamente la jugada física de una carta por parte de la mosca."""
        if not self.fly_cards or self.hand_over:
            return

        flat_table = [c["card_obj"] for round_c in self.table_cards for c in round_c]
        st = self.fly.encode_state(self.fly_cards, self.muestra, flat_table, self.trick_idx, self.truco_level, True)
        self.latest_telemetry = self.fly.get_detailed_telemetry(st)

        fly_idx = self.fly.select_card_action(self.fly_cards, st)
        f_card = self.fly_cards.pop(fly_idx)
        self.table_cards[self.trick_idx].append({
            "player": 1,
            "card": f_card,
            "card_obj": f_card,
            "label": card_label(f_card, self.muestra)
        })
        self.add_log(f"Mosca jugó: {card_label(f_card, self.muestra)}", "card")

    def _fly_turn(self):
        """Ejecuta el turno completo de la mosca: evalúa apuestas primero, y si no canta, tira carta."""
        if self.hand_over or self.match_over or not self.fly_cards or self.pending_prompt:
            return

        flat_table = [c["card_obj"] for round_c in self.table_cards for c in round_c]
        st = self.fly.encode_state(self.fly_cards, self.muestra, flat_table, self.trick_idx, self.truco_level, True)
        self.latest_telemetry = self.fly.get_detailed_telemetry(st)

        # 1. Evaluar si la mosca decide subir o cantar Truco
        can_raise = (self.truco_level == 0) or (self.truco_level in [1, 2] and self.canto_holder == 1)
        if can_raise:
            target_level = None
            if self.truco_level == 0 and self.fly.decide_truco(st):
                target_level = 1
            elif self.truco_level == 1 and self.fly.decide_retruco(st):
                target_level = 2
            elif self.truco_level == 2 and self.fly.decide_vale_cuatro(st):
                target_level = 3

            if target_level is not None:
                call_names = {1: "¡¡TRUCO!!", 2: "¡¡RE-TRUCO!!", 3: "¡¡VALE CUATRO!!"}
                pts_val = TRUCO_VALUES[target_level]
                prev_val = TRUCO_REJECT_POINTS.get(target_level, 1)
                self.add_log(f"🐝 La mosca zumba con agresividad y canta: {call_names[target_level]} (Vale {pts_val} pts)", "truco")

                opts = [
                    {"id": "quiero", "label": f"Quiero ({pts_val} puntos)"},
                    {"id": "no_quiero", "label": f"No quiero (ceder {prev_val} pt)"}
                ]
                if target_level == 1:
                    opts.append({"id": "retruco", "label": "¡¡QUIERO RE-TRUCO!! (3 pts)"})
                elif target_level == 2:
                    opts.append({"id": "vale_4", "label": "¡¡QUIERO VALE CUATRO!! (4 pts)"})

                self.pending_prompt = {
                    "type": "respond_truco",
                    "target_level": target_level,
                    "title": f"La mosca cantó {call_names[target_level]}",
                    "options": opts
                }
                # Queda a la espera de respuesta del humano sin tirar carta todavía
                return

        # 2. Si no cantó, la mosca juega carta
        self._fly_play_card_only()

        # 3. Si la baza ahora tiene 2 cartas, resolver baza
        if len(self.table_cards[self.trick_idx]) == 2:
            self._resolve_trick()
        else:
            # La mosca abrió la baza, ahora el turno es del humano
            self.current_turn = 0

    def _resolve_trick(self):
        """Resuelve el resultado de la baza actual de forma 100% segura contra StopIteration."""
        baza = self.table_cards[self.trick_idx]
        if len(baza) != 2:
            return

        by_player = {c["player"]: c["card_obj"] for c in baza}
        if 0 not in by_player or 1 not in by_player:
            return

        c_hum = by_player[0]
        c_fly = by_player[1]

        p_h = card_power(c_hum, self.muestra)
        p_f = card_power(c_fly, self.muestra)

        if p_h > p_f:
            self.trick_history.append(0)
            self.leader = 0
            self.add_log(f">> Ganaste la Baza #{self.trick_idx + 1}.", "win")
            self.fly.apply_dopamine_reward(-0.4)
        elif p_f > p_h:
            self.trick_history.append(1)
            self.leader = 1
            self.add_log(f">> La mosca mató tu carta y ganó la Baza #{self.trick_idx + 1}.", "loss")
            self.fly.apply_dopamine_reward(0.6)
        else:
            self.trick_history.append(-1)
            self.leader = 0 if self.human_is_hand else 1
            self.add_log(f">> ¡Parda en la Baza #{self.trick_idx + 1}!", "info")

        val_pts = TRUCO_VALUES[self.truco_level]
        dop_mult = 1.0 + 0.5 * self.truco_level

        # Verificar si la mano terminó (Reglas Oficiales Uruguayas)
        if self.trick_history.count(0) == 2:
            self.human_score += val_pts
            self.add_log(f"🏆 ¡¡GANASTE LA MANO DE TRUCO!! (+{val_pts} puntos)", "win")
            self.fly.apply_dopamine_reward(-1.5 * dop_mult)
            self.hand_over = True
        elif self.trick_history.count(1) == 2:
            self.fly_score += val_pts
            self.add_log(f"⚡ La mosca te ganó la mano de Truco. (+{val_pts} puntos)", "loss")
            self.fly.apply_dopamine_reward(2.0 * dop_mult)
            self.hand_over = True
        elif len(self.trick_history) == 2 and self.trick_history[0] == -1:
            if self.trick_history[1] == 0:
                self.human_score += val_pts
                self.add_log(f"🏆 Desempardaste en segunda: ¡Ganaste el Truco! (+{val_pts} pts)", "win")
                self.fly.apply_dopamine_reward(-1.5 * dop_mult)
                self.hand_over = True
            elif self.trick_history[1] == 1:
                self.fly_score += val_pts
                self.add_log(f"⚡ La mosca desempardó en segunda: Gana el Truco. (+{val_pts} pts)", "loss")
                self.fly.apply_dopamine_reward(2.0 * dop_mult)
                self.hand_over = True
        elif len(self.trick_history) == 3:
            h_wins = self.trick_history.count(0)
            f_wins = self.trick_history.count(1)
            if h_wins > f_wins:
                self.human_score += val_pts
                self.add_log(f"🏆 Ganaste en tercera mano (+{val_pts} pts)", "win")
                self.fly.apply_dopamine_reward(-1.5 * dop_mult)
            elif f_wins > h_wins:
                self.fly_score += val_pts
                self.add_log(f"⚡ La mosca ganó en tercera mano (+{val_pts} pts)", "loss")
                self.fly.apply_dopamine_reward(2.0 * dop_mult)
            else:
                if self.human_is_hand:
                    self.human_score += val_pts
                    self.add_log(f"🏆 Triple parda. Ganas por mano (+{val_pts} pts)", "win")
                    self.fly.apply_dopamine_reward(-1.0 * dop_mult)
                else:
                    self.fly_score += val_pts
                    self.add_log(f"⚡ Triple parda. Mosca gana por mano (+{val_pts} pts)", "loss")
                    self.fly.apply_dopamine_reward(1.5 * dop_mult)
            self.hand_over = True

        if not self.hand_over:
            self.trick_idx += 1
            self.current_turn = self.leader
            # Si el nuevo líder es la mosca, le toca abrir la siguiente baza
            if self.leader == 1:
                self._fly_turn()

    def get_state(self) -> Dict[str, Any]:
        """Serializa el estado completo para el frontend web."""
        piezas = get_effective_piezas(self.muestra) if self.muestra else []
        can_play = (self.current_turn == 0 and not self.pending_prompt and not self.hand_over and not self.match_over)

        # Cálculo de tantos en mano para información inmediata del jugador
        h_has_flor = has_flor(self.human_cards, self.muestra) if self.muestra and self.human_cards else False
        h_flor_pts = calculate_flor_points(self.human_cards, self.muestra) if h_has_flor else 0
        h_envido_pts = calculate_envido(self.human_cards, self.muestra) if self.muestra and self.human_cards else 0

        return {
            "hand_num": self.hand_num,
            "human_score": self.human_score,
            "fly_score": self.fly_score,
            "target_score": self.target_score,
            "match_over": self.human_score >= self.target_score or self.fly_score >= self.target_score,
            "winner": "Humano" if self.human_score >= self.target_score else ("Mosca" if self.fly_score >= self.target_score else None),
            "human_is_hand": self.human_is_hand,
            "current_turn": self.current_turn,
            "can_human_play_card": can_play,
            "turn_text": "🟢 Tu turno de jugar carta" if can_play else ("🟡 Turno de la mosca..." if not self.hand_over else "Mano finalizada"),
            "hand_over": self.hand_over,
            "trick_idx": self.trick_idx,
            "trick_history": self.trick_history,
            "truco_level": self.truco_level,
            "truco_level_name": TRUCO_NAMES[self.truco_level],
            "canto_holder": self.canto_holder,
            "can_human_call_truco": (self.truco_level == 0 and not self.pending_prompt and not self.hand_over and self.current_turn == 0),
            "can_human_call_retruco": (self.truco_level == 1 and self.canto_holder == 0 and not self.pending_prompt and not self.hand_over and self.current_turn == 0),
            "can_human_call_vale4": (self.truco_level == 2 and self.canto_holder == 0 and not self.pending_prompt and not self.hand_over and self.current_turn == 0),
            "can_human_call_envido": (self.envido_state == "NONE" and self.trick_idx == 0 and not self.pending_prompt and not self.hand_over and not any(c["player"] == 0 for c in self.table_cards[0]) and self.current_turn == 0),
            "envido_state": self.envido_state,
            "human_envido_points": h_envido_pts,
            "human_has_flor": h_has_flor,
            "human_flor_points": h_flor_pts,
            "pending_prompt": self.pending_prompt,
            "muestra": {
                "number": self.muestra.number,
                "suit": self.muestra.suit,
                "label": card_label(self.muestra, self.muestra),
                "piezas": piezas
            } if self.muestra else None,
            "human_cards": [
                {
                    "idx": i,
                    "number": c.number,
                    "suit": c.suit,
                    "label": card_label(c, self.muestra),
                    "power": card_power(c, self.muestra)
                }
                for i, c in enumerate(self.human_cards)
            ],
            "fly_card_count": len(self.fly_cards),
            "table_cards": [
                [
                    {
                        "player": c["player"],
                        "number": c["card"].number,
                        "suit": c["card"].suit,
                        "label": c["label"]
                    }
                    for c in round_c
                ]
                for round_c in self.table_cards
            ],
            "log": self.log[-15:],
            "telemetry": self.latest_telemetry,
            "kc_coords": self.kc_coords
        }

# Instancia global de la sesión
game_session = GameSession()

# Schemas de solicitudes
class PlayCardReq(BaseModel):
    card_idx: int

class RespondTrucoReq(BaseModel):
    action: str # "quiero", "no_quiero", "retruco", "vale_4"

class RespondEnvidoReq(BaseModel):
    action: str # "quiero", "no_quiero"

class TrainReq(BaseModel):
    episodes: int = 200
    lr: float = 0.015

class ConfigReq(BaseModel):
    target_score: int

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/game/state")
def get_state():
    return game_session.get_state()

@app.post("/api/game/config")
def update_config(req: ConfigReq):
    if req.target_score in [15, 30]:
        game_session.target_score = req.target_score
        game_session.add_log(f"⚙️ Chico configurado a {req.target_score} puntos.", "header")
    return game_session.get_state()

@app.post("/api/game/new_hand")
def new_hand():
    game_session.start_new_hand()
    return game_session.get_state()

@app.post("/api/game/new_match")
def new_match():
    global game_session
    game_session = GameSession()
    return game_session.get_state()

@app.post("/api/game/play_card")
def play_card(req: PlayCardReq):
    if game_session.hand_over or game_session.match_over:
        return game_session.get_state()
    if game_session.pending_prompt:
        raise HTTPException(status_code=400, detail="Debes responder la apuesta pendiente primero.")
    if game_session.current_turn != 0:
        raise HTTPException(status_code=400, detail="No es tu turno de jugar carta.")
    if any(c["player"] == 0 for c in game_session.table_cards[game_session.trick_idx]):
        raise HTTPException(status_code=400, detail="Ya jugaste una carta en esta baza.")
    if req.card_idx < 0 or req.card_idx >= len(game_session.human_cards):
        raise HTTPException(status_code=400, detail="Índice de carta inválido.")

    h_card = game_session.human_cards.pop(req.card_idx)
    game_session.table_cards[game_session.trick_idx].append({
        "player": 0,
        "card": h_card,
        "card_obj": h_card,
        "label": card_label(h_card, game_session.muestra)
    })
    game_session.add_log(f"Tiraste: {card_label(h_card, game_session.muestra)}", "card")

    # Si el humano abrió la baza (len == 1), le toca responder a la mosca
    if len(game_session.table_cards[game_session.trick_idx]) == 1:
        game_session.current_turn = 1
        game_session._fly_turn()
    else:
        # La mosca ya había abierto y el humano jugó la segunda carta
        game_session._resolve_trick()

    return game_session.get_state()

@app.post("/api/game/call_truco")
def call_truco():
    """El humano inicia el canto de Truco (nivel 1)."""
    if game_session.truco_level != 0 or game_session.pending_prompt or game_session.current_turn != 0:
        raise HTTPException(status_code=400, detail="No puedes cantar Truco en este momento.")

    game_session.add_log("✋ Tú golpeas la mesa: ¡¡TRUCO!! (Vale 2 puntos)", "truco")
    flat_table = [c["card_obj"] for round_c in game_session.table_cards for c in round_c]
    st = game_session.fly.encode_state(game_session.fly_cards, game_session.muestra, flat_table, game_session.trick_idx, 1, True)
    game_session.latest_telemetry = game_session.fly.get_detailed_telemetry(st)

    f_resp = game_session.fly.decide_truco_response(st, 1)

    if f_resp == "no_quiero":
        game_session.add_log("🐝 La mosca se va al mazo ante tu Truco. ¡Sumas 1 punto!", "win")
        game_session.human_score += 1
        game_session.fly.apply_dopamine_reward(-0.8)
        game_session.hand_over = True
    elif f_resp == "retruco":
        game_session.add_log("🐝 ¡La mosca retruca desafiante: ¡¡QUIERO RE-TRUCO!! 🔥 (Vale 3 pts)", "truco")
        game_session.truco_level = 2
        game_session.pending_prompt = {
            "type": "respond_truco",
            "target_level": 2,
            "title": "La mosca cantó RE-TRUCO",
            "options": [
                {"id": "quiero", "label": "Quiero (vale 3 puntos)"},
                {"id": "no_quiero", "label": "No quiero (ceder 1 punto)"},
                {"id": "vale_4", "label": "¡¡QUIERO VALE CUATRO!! (4 puntos)"}
            ]
        }
    else:
        game_session.add_log("🐝 La mosca responde: ¡¡QUIERO EL TRUCO!!", "truco")
        game_session.truco_level = 1
        game_session.canto_holder = 1 # Mosca tiene el derecho de cantar Re-truco luego

    return game_session.get_state()

@app.post("/api/game/call_retruco")
def call_retruco():
    """El humano canta Re-truco (nivel 2)."""
    if game_session.truco_level != 1 or game_session.canto_holder != 0 or game_session.pending_prompt or game_session.current_turn != 0:
        raise HTTPException(status_code=400, detail="No tienes derecho a cantar Re-truco.")

    game_session.add_log("🔥 Tú cantas: ¡¡RE-TRUCO!! (Vale 3 puntos)", "truco")
    flat_table = [c["card_obj"] for round_c in game_session.table_cards for c in round_c]
    st = game_session.fly.encode_state(game_session.fly_cards, game_session.muestra, flat_table, game_session.trick_idx, 2, True)
    game_session.latest_telemetry = game_session.fly.get_detailed_telemetry(st)

    f_resp = game_session.fly.decide_truco_response(st, 2)

    if f_resp == "no_quiero":
        game_session.add_log("🐝 La mosca no quiere el Re-truco. ¡Sumas 2 puntos!", "win")
        game_session.human_score += 2
        game_session.fly.apply_dopamine_reward(-1.2)
        game_session.hand_over = True
    elif f_resp == "vale_4":
        game_session.add_log("🐝 ¡La mosca no se achica y desafía: ¡¡QUIERO VALE CUATRO!! 💥 (Vale 4 pts)", "truco")
        game_session.truco_level = 3
        game_session.pending_prompt = {
            "type": "respond_truco",
            "target_level": 3,
            "title": "La mosca cantó VALE 4",
            "options": [
                {"id": "quiero", "label": "Quiero (vale 4 puntos)"},
                {"id": "no_quiero", "label": "No quiero (ceder 2 puntos)"}
            ]
        }
    else:
        game_session.add_log("🐝 La mosca responde: ¡¡QUIERO EL RE-TRUCO!!", "truco")
        game_session.truco_level = 2
        game_session.canto_holder = 1

    return game_session.get_state()

@app.post("/api/game/call_vale4")
def call_vale4():
    """El humano canta Vale 4 (nivel 3)."""
    if game_session.truco_level != 2 or game_session.canto_holder != 0 or game_session.pending_prompt or game_session.current_turn != 0:
        raise HTTPException(status_code=400, detail="No tienes derecho a cantar Vale 4.")

    game_session.add_log("💥 Tú golpeas la mesa: ¡¡VALE CUATRO!! (Vale 4 puntos)", "truco")
    flat_table = [c["card_obj"] for round_c in game_session.table_cards for c in round_c]
    st = game_session.fly.encode_state(game_session.fly_cards, game_session.muestra, flat_table, game_session.trick_idx, 3, True)
    game_session.latest_telemetry = game_session.fly.get_detailed_telemetry(st)

    f_resp = game_session.fly.decide_truco_response(st, 3)

    if f_resp == "no_quiero":
        game_session.add_log("🐝 La mosca se achica ante tu Vale 4. ¡Sumas 3 puntos!", "win")
        game_session.human_score += 3
        game_session.fly.apply_dopamine_reward(-1.8)
        game_session.hand_over = True
    else:
        game_session.add_log("🐝 ¡La mosca hace vibrar sus alas: ¡¡QUIERO EL VALE CUATRO!!", "truco")
        game_session.truco_level = 3
        game_session.canto_holder = None

    return game_session.get_state()

@app.post("/api/game/respond_truco")
def respond_truco(req: RespondTrucoReq):
    """El humano responde a un canto o subida de la mosca."""
    if not game_session.pending_prompt or game_session.pending_prompt.get("type") != "respond_truco":
        raise HTTPException(status_code=400, detail="No hay apuesta de Truco pendiente.")

    target_lvl = game_session.pending_prompt["target_level"]
    game_session.pending_prompt = None

    if req.action == "no_quiero":
        surrender_pts = TRUCO_REJECT_POINTS.get(target_lvl, 1)
        if getattr(game_session.fly, "is_bluffing", False):
            fly_cards_labels = ", ".join([card_label(c, game_session.muestra) for c in game_session.fly_cards])
            game_session.add_log(f"🃏 ¡LA MOSCA TE METIÓ UN FAROL! Te robó {surrender_pts} pt(s) mintiendo con: {fly_cards_labels}", "bluff")
            game_session.fly.apply_dopamine_reward(2.5)
        else:
            game_session.add_log(f"Te fuiste al mazo. La mosca suma {surrender_pts} punto(s).", "loss")
            game_session.fly.apply_dopamine_reward(1.0)
        game_session.fly_score += surrender_pts
        game_session.hand_over = True
        return game_session.get_state()

    elif req.action == "quiero":
        game_session.truco_level = target_lvl
        game_session.canto_holder = 0 # Humano tiene derecho a subir en la siguiente baza
        if getattr(game_session.fly, "is_bluffing", False):
            game_session.add_log("⚡ ¡Aceptaste el desafío! La mosca estaba faroleando e intentará defenderse.", "truco")
            game_session.fly.apply_dopamine_reward(-1.5)
        else:
            game_session.add_log(f"Tú dijiste: ¡QUIERO! (Mano vale {TRUCO_VALUES[target_lvl]} puntos)", "truco")

        # Reanudación obligatoria de la carta de la mosca:
        baza = game_session.table_cards[game_session.trick_idx]
        if not any(c["player"] == 1 for c in baza):
            game_session._fly_play_card_only()
            if len(baza) == 2:
                game_session._resolve_trick()
            else:
                # La mosca abrió la baza, ahora le toca al humano
                game_session.current_turn = 0
        return game_session.get_state()

    elif req.action == "retruco":
        game_session.add_log("🔥 Tú contra-atacas: ¡¡QUIERO RE-TRUCO!!", "truco")
        flat_table = [c["card_obj"] for round_c in game_session.table_cards for c in round_c]
        st = game_session.fly.encode_state(game_session.fly_cards, game_session.muestra, flat_table, game_session.trick_idx, 2, True)
        f_ans = game_session.fly.decide_truco_response(st, 2)
        if f_ans == "no_quiero":
            game_session.add_log("🐝 La mosca se achica ante tu Re-truco. ¡Sumas 1 punto!", "win")
            game_session.human_score += 1
            game_session.fly.apply_dopamine_reward(-1.0)
            game_session.hand_over = True
        elif f_ans == "vale_4":
            game_session.add_log("🐝 ¡La mosca redobla: ¡¡QUIERO VALE CUATRO!! 💥", "truco")
            game_session.truco_level = 3
            game_session.pending_prompt = {
                "type": "respond_truco",
                "target_level": 3,
                "title": "La mosca cantó VALE 4",
                "options": [
                    {"id": "quiero", "label": "Quiero (vale 4 puntos)"},
                    {"id": "no_quiero", "label": "No quiero (ceder 2 puntos)"}
                ]
            }
        else:
            game_session.add_log("🐝 La mosca responde: ¡¡QUIERO EL RE-TRUCO!!", "truco")
            game_session.truco_level = 2
            game_session.canto_holder = 1
            baza = game_session.table_cards[game_session.trick_idx]
            if not any(c["player"] == 1 for c in baza):
                game_session._fly_play_card_only()
                if len(baza) == 2:
                    game_session._resolve_trick()
                else:
                    game_session.current_turn = 0

    elif req.action == "vale_4":
        game_session.add_log("💥 Tú redoblas con todo: ¡¡QUIERO VALE CUATRO!!", "truco")
        flat_table = [c["card_obj"] for round_c in game_session.table_cards for c in round_c]
        st = game_session.fly.encode_state(game_session.fly_cards, game_session.muestra, flat_table, game_session.trick_idx, 3, True)
        f_ans = game_session.fly.decide_truco_response(st, 3)
        if f_ans == "no_quiero":
            game_session.add_log("🐝 La mosca no quiso el Vale 4. ¡Sumas 2 puntos!", "win")
            game_session.human_score += 2
            game_session.fly.apply_dopamine_reward(-1.5)
            game_session.hand_over = True
        else:
            game_session.add_log("🐝 La mosca responde con valentía: ¡¡QUIERO EL VALE CUATRO!!", "truco")
            game_session.truco_level = 3
            game_session.canto_holder = None
            baza = game_session.table_cards[game_session.trick_idx]
            if not any(c["player"] == 1 for c in baza):
                game_session._fly_play_card_only()
                if len(baza) == 2:
                    game_session._resolve_trick()
                else:
                    game_session.current_turn = 0

    return game_session.get_state()

@app.post("/api/game/call_envido")
def call_envido():
    """El humano canta Envido al inicio de la primera baza."""
    if game_session.envido_state != "NONE" or game_session.trick_idx != 0 or any(c["player"] == 0 for c in game_session.table_cards[0]) or game_session.current_turn != 0:
        raise HTTPException(status_code=400, detail="No puedes cantar Envido ahora.")

    game_session.add_log("✋ Tú anuncias: ¡¡ENVIDO!! (2 puntos)", "truco")
    st = game_session.fly.encode_state(game_session.fly_cards, game_session.muestra, [], 0, 0, True)
    game_session.latest_telemetry = game_session.fly.get_detailed_telemetry(st)

    if game_session.fly.decide_accept_envido(st):
        h_env = calculate_envido(game_session.human_cards, game_session.muestra)
        f_env = calculate_envido(game_session.fly_cards, game_session.muestra)
        game_session.add_log(f"🐝 La mosca dice: ¡QUIERO! | Tus tantos: {h_env} vs Mosca: {f_env}", "truco")
        if h_env > f_env:
            game_session.human_score += 2
            game_session.add_log(">> ¡Ganaste el Envido! (+2 puntos)", "win")
            game_session.fly.apply_dopamine_reward(-1.0)
        elif f_env > h_env:
            game_session.fly_score += 2
            game_session.add_log(">> La mosca gana el Envido (+2 puntos)", "loss")
            game_session.fly.apply_dopamine_reward(1.5)
        else:
            # Empate: gana el que es mano
            if game_session.human_is_hand:
                game_session.human_score += 2
                game_session.add_log(">> Empate en Envido: ¡Ganas por ser mano! (+2 pts)", "win")
                game_session.fly.apply_dopamine_reward(-1.0)
            else:
                game_session.fly_score += 2
                game_session.add_log(">> Empate en Envido: Mosca gana por ser mano (+2 pts)", "loss")
                game_session.fly.apply_dopamine_reward(1.5)
    else:
        game_session.add_log("🐝 La mosca responde: No quiero. Sumas 1 punto.", "win")
        game_session.human_score += 1

    game_session.envido_state = "RESOLVED"
    return game_session.get_state()

@app.post("/api/game/respond_envido")
def respond_envido(req: RespondEnvidoReq):
    """El humano responde al Envido cantado por la mosca."""
    if not game_session.pending_prompt or game_session.pending_prompt.get("type") != "respond_envido":
        raise HTTPException(status_code=400, detail="No hay Envido pendiente.")

    game_session.pending_prompt = None

    if req.action == "quiero":
        h_env = calculate_envido(game_session.human_cards, game_session.muestra)
        f_env = calculate_envido(game_session.fly_cards, game_session.muestra)
        game_session.add_log(f"Tú dices: ¡QUIERO! | Tus tantos: {h_env} vs Mosca: {f_env}", "truco")
        if h_env > f_env:
            game_session.human_score += 2
            game_session.add_log(">> ¡Ganaste el Envido! (+2 puntos)", "win")
            game_session.fly.apply_dopamine_reward(-1.0)
        elif f_env > h_env:
            game_session.fly_score += 2
            game_session.add_log(">> La mosca gana el Envido (+2 puntos)", "loss")
            game_session.fly.apply_dopamine_reward(1.5)
        else:
            # Empate: gana el que es mano
            if game_session.human_is_hand:
                game_session.human_score += 2
                game_session.add_log(">> Empate en Envido: ¡Ganas por ser mano! (+2 pts)", "win")
                game_session.fly.apply_dopamine_reward(-1.0)
            else:
                game_session.fly_score += 2
                game_session.add_log(">> Empate en Envido: Mosca gana por ser mano (+2 pts)", "loss")
                game_session.fly.apply_dopamine_reward(1.5)
    else:
        f_env = calculate_envido(game_session.fly_cards, game_session.muestra)
        if f_env < 24:
            game_session.add_log(f"🃏 ¡LA MOSCA TE METIÓ UN FAROL! Cantó Envido con solo {f_env} tantos y te robó 1 punto.", "bluff")
            game_session.fly.apply_dopamine_reward(2.0)
        else:
            game_session.add_log("Tú dices: No quiero. La mosca suma 1 punto.", "loss")
            game_session.fly.apply_dopamine_reward(0.8)
        game_session.fly_score += 1

    game_session.envido_state = "RESOLVED"

    # Si la mosca era mano y abrió con envido, ahora le corresponde tirar la primera carta de la mano
    if not game_session.human_is_hand and len(game_session.table_cards[0]) == 0:
        game_session._fly_turn()

    return game_session.get_state()

@app.post("/api/game/fold")
def fold():
    """El humano se va al mazo."""
    if game_session.hand_over:
        return game_session.get_state()

    p_loss = TRUCO_VALUES.get(game_session.truco_level, 1)
    if getattr(game_session.fly, "is_bluffing", False):
        fly_cards_labels = ", ".join([card_label(c, game_session.muestra) for c in game_session.fly_cards])
        game_session.add_log(f"🃏 ¡LA MOSCA TE METIÓ UN FAROL! Te fuiste al mazo y te robó {p_loss} pt(s) mintiendo con: {fly_cards_labels}", "bluff")
        game_session.fly.apply_dopamine_reward(2.5)
    else:
        game_session.add_log(f"Te fuiste al mazo. La mosca suma {p_loss} punto(s).", "loss")
        game_session.fly.apply_dopamine_reward(1.0)
    game_session.fly_score += p_loss
    game_session.hand_over = True
    return game_session.get_state()

@app.post("/api/train/start")
def train_fly(req: TrainReq):
    """Entrena el cerebro de la mosca por refuerzo dopaminérgico."""
    res = train_fly_session(game_session.fly, episodes=req.episodes, lr=req.lr)
    game_session.fly.save_model("data/fly_truco_model.npz")
    game_session.add_log(f"🎓 Entrenamiento completado: {req.episodes} manos. WinRate: {res['final_win_rate']}%.", "header")
    return res

@app.post("/api/brain/reset")
def reset_brain():
    """Reinicia a los pesos biológicos originales del conectoma de FlyEM."""
    game_session.fly.W = game_session.fly.W_bio.copy()
    game_session.fly.dopamine_history = []
    game_session.add_log("🧬 Pesos restaurados al conectoma biológico original vírgen.", "header")
    return {"status": "ok"}
