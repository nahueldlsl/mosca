"""
Entrenamiento por Auto-Juego (Self-Play / AlphaFly) para Truco Uruguayo.
Dos cerebros de mosca (Mushroom Body) compiten entre sí durante miles de manos,
descubriendo estrategias de blofeo y contrablofeo hacia el Equilibrio de Nash.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import argparse
import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from truco_engine import (
    Card, create_deck, card_power, calculate_envido, 
    get_effective_piezas, has_flor, calculate_flor_points
)
from fly_brain_agent import FlyBrainTrucoAgent

def simulate_self_play_hand(agent_0: FlyBrainTrucoAgent, agent_1: FlyBrainTrucoAgent, p0_is_hand: bool) -> Tuple[int, int]:
    deck = create_deck()
    random.shuffle(deck)
    muestra = deck.pop()

    cards_0 = [deck.pop() for _ in range(3)]
    cards_1 = [deck.pop() for _ in range(3)]

    pts_0 = 0
    pts_1 = 0

    # 1. FLOR
    flor_0 = has_flor(cards_0, muestra)
    flor_1 = has_flor(cards_1, muestra)
    flor_contested = False

    if flor_0 and flor_1:
        f0 = calculate_flor_points(cards_0, muestra)
        f1 = calculate_flor_points(cards_1, muestra)
        if f0 >= f1:
            pts_0 += 3
            agent_0.apply_dopamine_reward(2.0)
            agent_1.apply_dopamine_reward(-1.0)
        else:
            pts_1 += 3
            agent_1.apply_dopamine_reward(2.0)
            agent_0.apply_dopamine_reward(-1.0)
        flor_contested = True
    elif flor_0:
        pts_0 += 3
        agent_0.apply_dopamine_reward(2.0)
        flor_contested = True
    elif flor_1:
        pts_1 += 3
        agent_1.apply_dopamine_reward(2.0)
        flor_contested = True

    # 2. ENVIDO
    if not flor_contested:
        env_0 = calculate_envido(cards_0, muestra)
        env_1 = calculate_envido(cards_1, muestra)

        st_0 = agent_0.encode_state(cards_0, muestra, [], 0, 0, False)
        st_1 = agent_1.encode_state(cards_1, muestra, [], 0, 0, False)

        call_0 = agent_0.decide_envido(st_0)
        call_1 = agent_1.decide_envido(st_1)

        if p0_is_hand:
            if call_0:
                if agent_1.decide_envido(st_1):
                    # Ambos quieren
                    if env_0 >= env_1:
                        pts_0 += 2
                        agent_0.apply_dopamine_reward(1.5)
                        agent_1.apply_dopamine_reward(-1.2)
                    else:
                        pts_1 += 2
                        agent_1.apply_dopamine_reward(1.5)
                        agent_0.apply_dopamine_reward(-1.2)
                else:
                    pts_0 += 1
                    agent_0.apply_dopamine_reward(0.8)
            elif call_1:
                if agent_0.decide_envido(st_0):
                    if env_0 >= env_1:
                        pts_0 += 2
                        agent_0.apply_dopamine_reward(1.5)
                        agent_1.apply_dopamine_reward(-1.2)
                    else:
                        pts_1 += 2
                        agent_1.apply_dopamine_reward(1.5)
                        agent_0.apply_dopamine_reward(-1.2)
                else:
                    pts_1 += 1
                    agent_1.apply_dopamine_reward(0.8)
        else:
            if call_1:
                if agent_0.decide_envido(st_0):
                    if env_1 >= env_0:
                        pts_1 += 2
                        agent_1.apply_dopamine_reward(1.5)
                        agent_0.apply_dopamine_reward(-1.2)
                    else:
                        pts_0 += 2
                        agent_0.apply_dopamine_reward(1.5)
                        agent_1.apply_dopamine_reward(-1.2)
                else:
                    pts_1 += 1
                    agent_1.apply_dopamine_reward(0.8)
            elif call_0:
                if agent_1.decide_envido(st_1):
                    if env_0 > env_1:
                        pts_0 += 2
                        agent_0.apply_dopamine_reward(1.5)
                        agent_1.apply_dopamine_reward(-1.2)
                    else:
                        pts_1 += 2
                        agent_1.apply_dopamine_reward(1.5)
                        agent_0.apply_dopamine_reward(-1.2)
                else:
                    pts_0 += 1
                    agent_0.apply_dopamine_reward(0.8)

    # 3. TRUCO
    truco_lvl = 0
    table_hist: List[Card] = []
    trick_wins = []
    leader = 0 if p0_is_hand else 1

    for trick_idx in range(3):
        if not cards_0 or not cards_1:
            break

        # Canto de Truco
        if truco_lvl == 0:
            st_0 = agent_0.encode_state(cards_0, muestra, table_hist, trick_idx, 0, True)
            st_1 = agent_1.encode_state(cards_1, muestra, table_hist, trick_idx, 0, True)

            if agent_0.decide_truco(st_0):
                truco_lvl = 1
                if not agent_1.decide_accept_truco(st_1):
                    pts_0 += 1
                    agent_0.apply_dopamine_reward(1.0)
                    agent_1.apply_dopamine_reward(-0.5)
                    return pts_0, pts_1
            elif agent_1.decide_truco(st_1):
                truco_lvl = 1
                if not agent_0.decide_accept_truco(st_0):
                    pts_1 += 1
                    agent_1.apply_dopamine_reward(1.0)
                    agent_0.apply_dopamine_reward(-0.5)
                    return pts_0, pts_1

        # Baza de cartas
        if leader == 0:
            st_0 = agent_0.encode_state(cards_0, muestra, table_hist, trick_idx, truco_lvl, True)
            c0 = cards_0.pop(agent_0.select_card_action(cards_0, st_0))
            st_1 = agent_1.encode_state(cards_1, muestra, table_hist + [c0], trick_idx, truco_lvl, True)
            c1 = cards_1.pop(agent_1.select_card_action(cards_1, st_1))
            table_hist.extend([c0, c1])
            p0, p1 = card_power(c0, muestra), card_power(c1, muestra)
        else:
            st_1 = agent_1.encode_state(cards_1, muestra, table_hist, trick_idx, truco_lvl, True)
            c1 = cards_1.pop(agent_1.select_card_action(cards_1, st_1))
            st_0 = agent_0.encode_state(cards_0, muestra, table_hist + [c1], trick_idx, truco_lvl, True)
            c0 = cards_0.pop(agent_0.select_card_action(cards_0, st_0))
            table_hist.extend([c1, c0])
            p0, p1 = card_power(c0, muestra), card_power(c1, muestra)

        if p0 > p1:
            trick_wins.append(0)
            leader = 0
            agent_0.apply_dopamine_reward(0.5)
            agent_1.apply_dopamine_reward(-0.3)
        elif p1 > p0:
            trick_wins.append(1)
            leader = 1
            agent_1.apply_dopamine_reward(0.5)
            agent_0.apply_dopamine_reward(-0.3)
        else:
            trick_wins.append(-1)
            leader = 0 if p0_is_hand else 1

        val = 2 if truco_lvl >= 1 else 1
        if trick_wins.count(0) == 2 or (len(trick_wins) == 2 and trick_wins[0] == -1 and trick_wins[1] == 0):
            pts_0 += val
            agent_0.apply_dopamine_reward(2.0)
            agent_1.apply_dopamine_reward(-1.8)
            break
        elif trick_wins.count(1) == 2 or (len(trick_wins) == 2 and trick_wins[0] == -1 and trick_wins[1] == 1):
            pts_1 += val
            agent_1.apply_dopamine_reward(2.0)
            agent_0.apply_dopamine_reward(-1.8)
            break

    if len(trick_wins) == 3 and pts_0 == 0 and pts_1 == 0:
        val = 2 if truco_lvl >= 1 else 1
        w0 = trick_wins.count(0)
        w1 = trick_wins.count(1)
        if w0 > w1:
            pts_0 += val
            agent_0.apply_dopamine_reward(2.0)
            agent_1.apply_dopamine_reward(-1.8)
        elif w1 > w0:
            pts_1 += val
            agent_1.apply_dopamine_reward(2.0)
            agent_0.apply_dopamine_reward(-1.8)
        else:
            if p0_is_hand:
                pts_0 += val
            else:
                pts_1 += val

    return pts_0, pts_1

def train_self_play(episodes: int = 3000, save_path: str = "data/fly_truco_alpha_model.npz"):
    print("=" * 65)
    print("      ENTRENAMIENTO POR AUTO-JUEGO (Self-Play / AlphaFly)")
    print("  Co-Evolución Biológica de dos moscas en Truco Uruguayo")
    print("=" * 65)

    fly_a = FlyBrainTrucoAgent()
    fly_b = FlyBrainTrucoAgent()

    # Si ya existe modelo previo, cargarlo como punto de partida
    if os.path.exists("data/fly_truco_model.npz"):
        fly_a.load_model("data/fly_truco_model.npz")
        fly_b.load_model("data/fly_truco_model.npz")

    pbar = tqdm(range(episodes), desc="Auto-juego")
    pts_a_total = 0
    pts_b_total = 0

    for ep in pbar:
        p0_is_hand = (ep % 2 == 0)
        p_a, p_b = simulate_self_play_hand(fly_a, fly_b, p0_is_hand)
        pts_a_total += p_a
        pts_b_total += p_b

        # Cada 100 manos, compartir descubrimientos sinápticos (sincronizar el mejor cerebro)
        if (ep + 1) % 100 == 0:
            avg_w = (fly_a.W + fly_b.W) / 2.0
            fly_a.W = avg_w.copy()
            fly_b.W = avg_w.copy()
            pbar.set_postfix({"Pts A": pts_a_total, "Pts B": pts_b_total})

    # Guardar el modelo Alpha entrenado por auto-juego
    fly_a.save_model(save_path)
    # También actualizar el modelo principal
    fly_a.save_model("data/fly_truco_model.npz")

    print("\n" + "=" * 65)
    print(" AUTO-JUEGO COMPLETADO CON ÉXITO")
    print(f" Manos de co-evolución: {episodes:,}")
    print(f" Modelo de Élite AlphaFly guardado en: {save_path} y data/fly_truco_model.npz")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    args = parser.parse_args()
    train_self_play(args.episodes)
