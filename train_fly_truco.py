"""
Entrenamiento del cerebro de la mosca (FlyBrainTrucoAgent)
mediante refuerzo dopaminérgico jugando miles de manos de Truco Uruguayo.
Reglamento Oficial: Muestra, Piezas, Flor, Envido y Truco.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import random
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from truco_engine import (
    Card, create_deck, card_power, calculate_envido, 
    get_effective_piezas, has_flor, calculate_flor_points
)
from fly_brain_agent import FlyBrainTrucoAgent

class HeuristicBot:
    """Oponente con estrategia básica de Truco Uruguayo para entrenar a la mosca."""
    def __init__(self, name: str = "Bot Gaucho"):
        self.name = name

    def decide_envido(self, hand: List[Card], muestra: Card) -> bool:
        if has_flor(hand, muestra):
            return False # Tiene flor, no canta envido
        pts = calculate_envido(hand, muestra)
        return pts >= 28

    def decide_accept_envido(self, hand: List[Card], muestra: Card) -> bool:
        pts = calculate_envido(hand, muestra)
        return pts >= 27

    def select_card(self, hand: List[Card], muestra: Card, table_card: Optional[Card]) -> int:
        if not table_card:
            powers = [card_power(c, muestra) for c in hand]
            sorted_idx = np.argsort(powers)
            return int(sorted_idx[0])
        else:
            opp_pow = card_power(table_card, muestra)
            winning_cards = [(i, card_power(c, muestra)) for i, c in enumerate(hand) if card_power(c, muestra) > opp_pow]
            if winning_cards:
                winning_cards.sort(key=lambda x: x[1])
                return winning_cards[0][0]
            else:
                powers = [(i, card_power(c, muestra)) for i, c in enumerate(hand)]
                powers.sort(key=lambda x: x[1])
                return powers[0][0]

    def decide_truco(self, hand: List[Card], muestra: Card, trick_idx: int) -> bool:
        max_pow = max([card_power(c, muestra) for c in hand]) if hand else 0
        return max_pow >= 88 and random.random() < 0.6

    def decide_accept_truco(self, hand: List[Card], muestra: Card) -> bool:
        max_pow = max([card_power(c, muestra) for c in hand]) if hand else 0
        return max_pow >= 75 or random.random() < 0.25

    def decide_truco_response(self, hand: List[Card], muestra: Card, current_level: int) -> str:
        """
        Responde a Truco (1), Re-truco (2) o Vale 4 (3).
        Retorna: 'no_quiero', 'quiero', 'retruco' o 'vale_4'.
        """
        powers = sorted([card_power(c, muestra) for c in hand], reverse=True) if hand else [0]
        max_pow = powers[0]
        second_pow = powers[1] if len(powers) > 1 else 0

        if current_level == 1:  # Rival cantó Truco
            if max_pow >= 96 or (max_pow >= 89 and second_pow >= 80):
                return "retruco" if random.random() < 0.65 else "quiero"
            elif max_pow >= 75 or random.random() < 0.35:
                return "quiero"
            else:
                return "no_quiero"
        elif current_level == 2:  # Rival cantó Re-truco
            if max_pow >= 98:  # Pieza 2 o 4
                return "vale_4" if random.random() < 0.70 else "quiero"
            elif max_pow >= 88 or (max_pow >= 80 and random.random() < 0.40):
                return "quiero"
            else:
                return "no_quiero"
        elif current_level == 3:  # Rival cantó Vale 4
            if max_pow >= 96 or (max_pow >= 88 and random.random() < 0.50):
                return "quiero"
            else:
                return "no_quiero"
        return "quiero"

    def decide_raise(self, hand: List[Card], muestra: Card, current_level: int) -> Optional[str]:
        powers = sorted([card_power(c, muestra) for c in hand], reverse=True) if hand else [0]
        max_pow = powers[0]
        if current_level == 1 and max_pow >= 90 and random.random() < 0.5:
            return "retruco"
        elif current_level == 2 and max_pow >= 96 and random.random() < 0.6:
            return "vale_4"
        return None

def simulate_hand(fly: FlyBrainTrucoAgent, bot: HeuristicBot, fly_is_hand: bool) -> Tuple[int, int, float]:
    deck = create_deck()
    random.shuffle(deck)
    muestra = deck.pop()

    fly_cards = [deck.pop() for _ in range(3)]
    bot_cards = [deck.pop() for _ in range(3)]

    fly_pts = 0
    bot_pts = 0
    dopamine_earned = 0.0

    # 1. VERIFICACIÓN DE FLOR
    fly_has_flor = has_flor(fly_cards, muestra)
    bot_has_flor = has_flor(bot_cards, muestra)
    envido_played = False

    if fly_has_flor and bot_has_flor:
        f_flor = calculate_flor_points(fly_cards, muestra)
        b_flor = calculate_flor_points(bot_cards, muestra)
        if f_flor >= b_flor:
            fly_pts += 3
            fly.apply_dopamine_reward(2.0)
            dopamine_earned += 2.0
        else:
            bot_pts += 3
            fly.apply_dopamine_reward(-1.0)
            dopamine_earned -= 1.0
        envido_played = True
    elif fly_has_flor:
        fly_pts += 3
        fly.apply_dopamine_reward(2.0)
        dopamine_earned += 2.0
        envido_played = True
    elif bot_has_flor:
        bot_pts += 3
        envido_played = True

    # 2. FASE DE ENVIDO (Solo si ninguno tiene Flor)
    if not envido_played:
        fly_envido_val = calculate_envido(fly_cards, muestra)
        bot_envido_val = calculate_envido(bot_cards, muestra)

        state_envido = fly.encode_state(fly_cards, muestra, [], 0, 0, False)
        fly_wants_envido = fly.decide_envido(state_envido)

        if fly_is_hand:
            if fly_wants_envido:
                bot_accepts = bot.decide_accept_envido(bot_cards, muestra)
                if bot_accepts:
                    if fly_envido_val >= bot_envido_val:
                        fly_pts += 2
                        fly.apply_dopamine_reward(1.5)
                        dopamine_earned += 1.5
                    else:
                        bot_pts += 2
                        fly.apply_dopamine_reward(-1.2)
                        dopamine_earned -= 1.2
                else:
                    fly_pts += 1
                    fly.apply_dopamine_reward(0.8)
                    dopamine_earned += 0.8
            elif bot.decide_envido(bot_cards, muestra):
                if fly.decide_accept_envido(state_envido):
                    if fly_envido_val >= bot_envido_val:
                        fly_pts += 2
                        fly.apply_dopamine_reward(1.5)
                        dopamine_earned += 1.5
                    else:
                        bot_pts += 2
                        fly.apply_dopamine_reward(-1.2)
                        dopamine_earned -= 1.2
                else:
                    bot_pts += 1
                    fly.apply_dopamine_reward(-0.5)
                    dopamine_earned -= 0.5
        else:
            if bot.decide_envido(bot_cards, muestra):
                if fly.decide_accept_envido(state_envido):
                    if fly_envido_val > bot_envido_val:
                        fly_pts += 2
                        fly.apply_dopamine_reward(1.5)
                        dopamine_earned += 1.5
                    else:
                        bot_pts += 2
                        fly.apply_dopamine_reward(-1.2)
                        dopamine_earned -= 1.2
                else:
                    bot_pts += 1
                    fly.apply_dopamine_reward(-0.5)
                    dopamine_earned -= 0.5
            elif fly_wants_envido:
                if bot.decide_accept_envido(bot_cards, muestra):
                    if fly_envido_val > bot_envido_val:
                        fly_pts += 2
                        fly.apply_dopamine_reward(1.5)
                        dopamine_earned += 1.5
                    else:
                        bot_pts += 2
                        fly.apply_dopamine_reward(-1.2)
                        dopamine_earned -= 1.2
                else:
                    fly_pts += 1
                    fly.apply_dopamine_reward(0.8)
                    dopamine_earned += 0.8

    # 3. FASE DE TRUCO (3 BAZAS CON ESCALERA COMPLETA: TRUCO, RE-TRUCO, VALE 4)
    trick_wins = []
    table_history: List[Card] = []
    truco_level = 0
    canto_holder = None  # 0: fly, 1: bot (quién tiene derecho a cantar el próximo nivel)
    leader = 0 if fly_is_hand else 1

    for trick_idx in range(3):
        if not fly_cards or not bot_cards:
            break

        # A. Evaluación de Apuestas de Truco / Retruco / Vale 4
        # Si truco_level == 0, cualquiera puede iniciar Truco
        if truco_level == 0:
            st_fly = fly.encode_state(fly_cards, muestra, table_history, trick_idx, truco_level, True)
            fly_initiates = fly.decide_truco(st_fly)
            bot_initiates = bot.decide_truco(bot_cards, muestra, trick_idx) if not fly_initiates else False

            if fly_initiates:
                # Mosca canta Truco (Nivel 1)
                resp = bot.decide_truco_response(bot_cards, muestra, 1)
                if resp == "no_quiero":
                    fly_pts += 1
                    fly.apply_dopamine_reward(1.0)
                    return fly_pts, bot_pts, dopamine_earned + 1.0
                elif resp == "quiero":
                    truco_level = 1
                    canto_holder = 1  # Bot tiene derecho a cantar Re-truco
                elif resp == "retruco":
                    # Bot redobla a Re-truco (Nivel 2)
                    st_re = fly.encode_state(fly_cards, muestra, table_history, trick_idx, 2, True)
                    fly_resp = fly.decide_truco_response(st_re, 2)
                    if fly_resp == "no_quiero":
                        bot_pts += 1  # Puntos del nivel anterior
                        fly.apply_dopamine_reward(-0.8)
                        return fly_pts, bot_pts, dopamine_earned - 0.8
                    elif fly_resp == "quiero":
                        truco_level = 2
                        canto_holder = 0  # Mosca tiene derecho a cantar Vale 4
                    elif fly_resp == "vale_4":
                        # Mosca redobla a Vale 4 (Nivel 3)
                        b_v4 = bot.decide_truco_response(bot_cards, muestra, 3)
                        if b_v4 == "no_quiero":
                            fly_pts += 2
                            fly.apply_dopamine_reward(2.0)
                            return fly_pts, bot_pts, dopamine_earned + 2.0
                        else:
                            truco_level = 3
                            canto_holder = None

            elif bot_initiates:
                # Bot canta Truco (Nivel 1)
                st_resp = fly.encode_state(fly_cards, muestra, table_history, trick_idx, 1, True)
                fly_resp = fly.decide_truco_response(st_resp, 1)
                if fly_resp == "no_quiero":
                    bot_pts += 1
                    fly.apply_dopamine_reward(-0.5)
                    return fly_pts, bot_pts, dopamine_earned - 0.5
                elif fly_resp == "quiero":
                    truco_level = 1
                    canto_holder = 0  # Mosca tiene derecho a cantar Re-truco
                elif fly_resp == "retruco":
                    # Mosca redobla a Re-truco (Nivel 2)
                    b_resp = bot.decide_truco_response(bot_cards, muestra, 2)
                    if b_resp == "no_quiero":
                        fly_pts += 1
                        fly.apply_dopamine_reward(1.5)
                        return fly_pts, bot_pts, dopamine_earned + 1.5
                    elif b_resp == "quiero":
                        truco_level = 2
                        canto_holder = 1  # Bot tiene derecho a cantar Vale 4
                    elif b_resp == "vale_4":
                        # Bot redobla a Vale 4 (Nivel 3)
                        st_v4 = fly.encode_state(fly_cards, muestra, table_history, trick_idx, 3, True)
                        f_v4 = fly.decide_truco_response(st_v4, 3)
                        if f_v4 == "no_quiero":
                            bot_pts += 2
                            fly.apply_dopamine_reward(-1.5)
                            return fly_pts, bot_pts, dopamine_earned - 1.5
                        else:
                            truco_level = 3
                            canto_holder = None

        # Si ya se cantó truco (level 1 o 2), verificar si quien tiene el canto decide subir
        elif truco_level in [1, 2] and canto_holder is not None:
            if canto_holder == 0:
                st_raise = fly.encode_state(fly_cards, muestra, table_history, trick_idx, truco_level, True)
                raise_call = fly.decide_raise_truco(st_raise, truco_level)
                if raise_call == "retruco":
                    b_ans = bot.decide_truco_response(bot_cards, muestra, 2)
                    if b_ans == "no_quiero":
                        fly_pts += 1
                        fly.apply_dopamine_reward(1.5)
                        return fly_pts, bot_pts, dopamine_earned + 1.5
                    elif b_ans == "quiero":
                        truco_level = 2
                        canto_holder = 1
                    elif b_ans == "vale_4":
                        st_v4 = fly.encode_state(fly_cards, muestra, table_history, trick_idx, 3, True)
                        if fly.decide_truco_response(st_v4, 3) == "no_quiero":
                            bot_pts += 2
                            fly.apply_dopamine_reward(-1.5)
                            return fly_pts, bot_pts, dopamine_earned - 1.5
                        else:
                            truco_level = 3
                            canto_holder = None
                elif raise_call == "vale_4":
                    b_ans = bot.decide_truco_response(bot_cards, muestra, 3)
                    if b_ans == "no_quiero":
                        fly_pts += 2
                        fly.apply_dopamine_reward(2.5)
                        return fly_pts, bot_pts, dopamine_earned + 2.5
                    else:
                        truco_level = 3
                        canto_holder = None
            elif canto_holder == 1:
                b_raise = bot.decide_raise(bot_cards, muestra, truco_level)
                if b_raise == "retruco":
                    st_re = fly.encode_state(fly_cards, muestra, table_history, trick_idx, 2, True)
                    f_ans = fly.decide_truco_response(st_re, 2)
                    if f_ans == "no_quiero":
                        bot_pts += 1
                        fly.apply_dopamine_reward(-1.0)
                        return fly_pts, bot_pts, dopamine_earned - 1.0
                    elif f_ans == "quiero":
                        truco_level = 2
                        canto_holder = 0
                    elif f_ans == "vale_4":
                        if bot.decide_truco_response(bot_cards, muestra, 3) == "no_quiero":
                            fly_pts += 2
                            fly.apply_dopamine_reward(2.5)
                            return fly_pts, bot_pts, dopamine_earned + 2.5
                        else:
                            truco_level = 3
                            canto_holder = None
                elif b_raise == "vale_4":
                    st_v4 = fly.encode_state(fly_cards, muestra, table_history, trick_idx, 3, True)
                    if fly.decide_truco_response(st_v4, 3) == "no_quiero":
                        bot_pts += 2
                        fly.apply_dopamine_reward(-2.0)
                        return fly_pts, bot_pts, dopamine_earned - 2.0
                    else:
                        truco_level = 3
                        canto_holder = None

        # B. Jugar cartas
        if leader == 0:
            st_card = fly.encode_state(fly_cards, muestra, table_history, trick_idx, truco_level, True)
            c_idx = fly.select_card_action(fly_cards, st_card)
            fly_c = fly_cards.pop(c_idx)

            bot_c_idx = bot.select_card(bot_cards, muestra, fly_c)
            bot_c = bot_cards.pop(bot_c_idx)

            table_history.extend([fly_c, bot_c])
            pow_fly = card_power(fly_c, muestra)
            pow_bot = card_power(bot_c, muestra)
        else:
            bot_c_idx = bot.select_card(bot_cards, muestra, None)
            bot_c = bot_cards.pop(bot_c_idx)

            st_card = fly.encode_state(fly_cards, muestra, table_history + [bot_c], trick_idx, truco_level, True)
            c_idx = fly.select_card_action(fly_cards, st_card)
            fly_c = fly_cards.pop(c_idx)

            table_history.extend([bot_c, fly_c])
            pow_fly = card_power(fly_c, muestra)
            pow_bot = card_power(bot_c, muestra)

        if pow_fly > pow_bot:
            trick_wins.append(0)
            leader = 0
            fly.apply_dopamine_reward(0.5)
            dopamine_earned += 0.5
        elif pow_bot > pow_fly:
            trick_wins.append(1)
            leader = 1
            fly.apply_dopamine_reward(-0.3)
            dopamine_earned -= 0.3
        else:
            trick_wins.append(-1)
            leader = 0 if fly_is_hand else 1

        val_pts = {0: 1, 1: 2, 2: 3, 3: 4}[truco_level]
        dop_mult = 1.0 + 0.5 * truco_level

        if trick_wins.count(0) == 2:
            fly_pts += val_pts
            reward = 2.0 * dop_mult
            fly.apply_dopamine_reward(reward)
            dopamine_earned += reward
            break
        elif trick_wins.count(1) == 2:
            bot_pts += val_pts
            punishment = -1.8 * dop_mult
            fly.apply_dopamine_reward(punishment)
            dopamine_earned += punishment
            break
        elif len(trick_wins) == 2 and trick_wins[0] == -1:
            if trick_wins[1] == 0:
                fly_pts += val_pts
                reward = 2.0 * dop_mult
                fly.apply_dopamine_reward(reward)
                dopamine_earned += reward
                break
            elif trick_wins[1] == 1:
                bot_pts += val_pts
                punishment = -1.8 * dop_mult
                fly.apply_dopamine_reward(punishment)
                dopamine_earned += punishment
                break

    if len(trick_wins) == 3 and fly_pts == 0 and bot_pts == 0:
        val_pts = {0: 1, 1: 2, 2: 3, 3: 4}[truco_level]
        fly_w = trick_wins.count(0)
        bot_w = trick_wins.count(1)
        if fly_w > bot_w:
            fly_pts += val_pts
            fly.apply_dopamine_reward(2.0)
            dopamine_earned += 2.0
        elif bot_w > fly_w:
            bot_pts += val_pts
            fly.apply_dopamine_reward(-1.8)
            dopamine_earned -= 1.8
        else:
            if fly_is_hand:
                fly_pts += val_pts
                fly.apply_dopamine_reward(1.5)
            else:
                bot_pts += val_pts
                fly.apply_dopamine_reward(-1.5)

    return fly_pts, bot_pts, dopamine_earned

def train_fly_session(fly: FlyBrainTrucoAgent, episodes: int = 200, lr: float = 0.015, callback=None) -> dict:
    """
    Función de entrenamiento por lotes invocable desde scripts o desde el dashboard web en tiempo real.
    """
    fly.lr = lr
    bot = HeuristicBot()
    win_history = []
    dopamine_curves = []
    fly_total_pts = 0
    bot_total_pts = 0

    for ep in range(episodes):
        fly_is_hand = (ep % 2 == 0)
        f_pts, b_pts, dopa = simulate_hand(fly, bot, fly_is_hand)
        fly_total_pts += f_pts
        bot_total_pts += b_pts
        win = 1 if f_pts > b_pts else 0
        win_history.append(win)
        dopamine_curves.append(dopa)

        if callback and (ep % 10 == 0 or ep == episodes - 1):
            recent_win_rate = float(np.mean(win_history[-30:]) * 100) if win_history else 0.0
            callback({
                "episode": ep + 1,
                "total_episodes": episodes,
                "recent_win_rate": round(recent_win_rate, 1),
                "fly_total_pts": fly_total_pts,
                "bot_total_pts": bot_total_pts,
                "total_dopamine": round(float(np.sum(dopamine_curves)), 2)
            })

    return {
        "episodes": episodes,
        "fly_total_pts": fly_total_pts,
        "bot_total_pts": bot_total_pts,
        "final_win_rate": round(float(np.mean(win_history[-50:]) * 100) if len(win_history) >= 50 else float(np.mean(win_history) * 100), 1),
        "total_dopamine": round(float(np.sum(dopamine_curves)), 2),
        "win_history": win_history,
        "dopamine_curves": [round(d, 2) for d in dopamine_curves]
    }

def train(episodes: int = 600, save_path: str = "data/fly_truco_model.npz"):
    print("=" * 65)
    print(" INICIANDO ENTRENAMIENTO DE LA MOSCA EN TRUCO URUGUAYO [UY]")
    print(f" Episodios (manos): {episodes}")
    print(" Arquitectura: Mushroom Body Biológico (FlyEM Male CNS v1.0)")
    print("=" * 65)

    fly = FlyBrainTrucoAgent()
    bot = HeuristicBot()

    fly_total_pts = 0
    bot_total_pts = 0
    win_history = []
    dopamine_curves = []

    window = 50
    pbar = tqdm(range(episodes), desc="Manos jugadas")

    for ep in pbar:
        fly_is_hand = (ep % 2 == 0)
        f_pts, b_pts, dopa = simulate_hand(fly, bot, fly_is_hand)

        fly_total_pts += f_pts
        bot_total_pts += b_pts
        win = 1 if f_pts > b_pts else 0
        win_history.append(win)
        dopamine_curves.append(dopa)

        if len(win_history) >= window:
            recent_win_rate = np.mean(win_history[-window:]) * 100
            pbar.set_postfix({
                "WinRate (ult 50)": f"{recent_win_rate:.1f}%",
                "Total Mosca": fly_total_pts,
                "Total Bot": bot_total_pts
            })

    fly.save_model(save_path)

    # Gráfico de Aprendizaje
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    moving_avg = np.convolve(win_history, np.ones(window)/window, mode='valid') * 100
    plt.plot(moving_avg, color="#1f77b4", lw=2, label="Tasa de Victorias (%)")
    plt.axhline(50, color='gray', linestyle='--', label="Línea 50% (Paridad)")
    plt.title("Curva de Aprendizaje de la Mosca (Truco Uruguayo)")
    plt.xlabel("Mano")
    plt.ylabel("% Victorias (Media móvil 50 manos)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(dopamine_curves), color="#2ca02c", lw=2)
    plt.title("Dopamina Acumulada (PAM vs PPL1)")
    plt.xlabel("Mano")
    plt.ylabel("Dopamina Acumulada")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = "data/fly_truco_learning_curve.png"
    plt.savefig(chart_path, dpi=180)
    plt.close()

    print("\n" + "=" * 65)
    print(" ENTRENAMIENTO COMPLETADO")
    print(f" Puntos finales acumulados: Mosca {fly_total_pts} vs Bot {bot_total_pts}")
    print(f" Tasa final de victorias (últimas 100 manos): {np.mean(win_history[-100:])*100:.1f}%")
    print(f" Gráfico de aprendizaje guardado en: {chart_path}")
    print(f" Modelo cerebral guardado en: {save_path}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600, help="Número de manos a entrenar")
    parser.add_argument("--save-path", default="data/fly_truco_model.npz", help="Ruta de guardado")
    args = parser.parse_args()
    train(args.episodes, args.save_path)
