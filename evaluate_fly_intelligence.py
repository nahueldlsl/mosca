"""
Evaluador Científico de Inteligencia de la Mosca en Truco Uruguayo.
Realiza tests de ablación comparando:
1. Mosca Entrenada (Dopamina PAM/PPL1 & Self-Play)
2. Mosca Ingenua (Pesos biológicos vírgenes de FlyEM sin entrenar)
3. Agente Aleatorio (Decisiones puramente al azar)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from truco_engine import (
    Card, create_deck, card_power, calculate_envido, 
    get_effective_piezas, has_flor, calculate_flor_points
)
from fly_brain_agent import FlyBrainTrucoAgent

class RandomAgent:
    """Agente que juega totalmente al azar (línea base nula)."""
    def decide_envido(self, state_vec) -> bool:
        return random.random() < 0.4

    def decide_truco(self, state_vec) -> bool:
        return random.random() < 0.3

    def decide_accept_truco(self, state_vec) -> bool:
        return random.random() < 0.5

    def select_card_action(self, hand: List[Card], state_vec) -> int:
        return random.randint(0, len(hand) - 1)

class NaiveFlyAgent(FlyBrainTrucoAgent):
    """Mosca con los pesos biológicos de FlyEM sin ningún entrenamiento dopaminérgico."""
    def __init__(self):
        super().__init__()

def simulate_duel(agent_a, agent_b, hands: int = 1000) -> Tuple[int, int, int, float]:
    """Enfrenta a dos agentes a lo largo de N manos."""
    wins_a = 0
    wins_b = 0
    ties = 0

    for h_idx in range(hands):
        deck = create_deck()
        random.shuffle(deck)
        muestra = deck.pop()

        cards_a = [deck.pop() for _ in range(3)]
        cards_b = [deck.pop() for _ in range(3)]
        a_is_hand = (h_idx % 2 == 0)

        pts_a = 0
        pts_b = 0

        # 1. FLOR
        flor_a = has_flor(cards_a, muestra)
        flor_b = has_flor(cards_b, muestra)
        flor_contested = False

        if flor_a and flor_b:
            if calculate_flor_points(cards_a, muestra) >= calculate_flor_points(cards_b, muestra):
                pts_a += 3
            else:
                pts_b += 3
            flor_contested = True
        elif flor_a:
            pts_a += 3
            flor_contested = True
        elif flor_b:
            pts_b += 3
            flor_contested = True

        # 2. ENVIDO (si no hay flor)
        if not flor_contested:
            env_a = calculate_envido(cards_a, muestra)
            env_b = calculate_envido(cards_b, muestra)

            st_a = agent_a.encode_state(cards_a, muestra, [], 0, 0, False) if hasattr(agent_a, 'encode_state') else None
            st_b = agent_b.encode_state(cards_b, muestra, [], 0, 0, False) if hasattr(agent_b, 'encode_state') else None

            call_a = agent_a.decide_envido(st_a)
            call_b = agent_b.decide_envido(st_b)

            if a_is_hand:
                if call_a:
                    if call_b:
                        if env_a >= env_b:
                            pts_a += 2
                        else:
                            pts_b += 2
                    else:
                        pts_a += 1
                elif call_b:
                    if call_a:
                        if env_a >= env_b:
                            pts_a += 2
                        else:
                            pts_b += 2
                    else:
                        pts_b += 1
            else:
                if call_b:
                    if call_a:
                        if env_b >= env_a:
                            pts_b += 2
                        else:
                            pts_a += 2
                    else:
                        pts_b += 1
                elif call_a:
                    if call_b:
                        if env_a > env_b:
                            pts_a += 2
                        else:
                            pts_b += 2
                    else:
                        pts_a += 1

        # 3. TRUCO
        truco_lvl = 0
        table_hist = []
        trick_wins = []
        leader = 0 if a_is_hand else 1

        for trick_idx in range(3):
            if not cards_a or not cards_b:
                break

            if truco_lvl == 0:
                st_a = agent_a.encode_state(cards_a, muestra, table_hist, trick_idx, 0, True) if hasattr(agent_a, 'encode_state') else None
                st_b = agent_b.encode_state(cards_b, muestra, table_hist, trick_idx, 0, True) if hasattr(agent_b, 'encode_state') else None

                if agent_a.decide_truco(st_a):
                    truco_lvl = 1
                    if not agent_b.decide_accept_truco(st_b):
                        pts_a += 1
                        break
                elif agent_b.decide_truco(st_b):
                    truco_lvl = 1
                    if not agent_a.decide_accept_truco(st_a):
                        pts_b += 1
                        break

            if leader == 0:
                st_a = agent_a.encode_state(cards_a, muestra, table_hist, trick_idx, truco_lvl, True) if hasattr(agent_a, 'encode_state') else None
                c_a = cards_a.pop(agent_a.select_card_action(cards_a, st_a))
                st_b = agent_b.encode_state(cards_b, muestra, table_hist + [c_a], trick_idx, truco_lvl, True) if hasattr(agent_b, 'encode_state') else None
                c_b = cards_b.pop(agent_b.select_card_action(cards_b, st_b))
                table_hist.extend([c_a, c_b])
                pa, pb = card_power(c_a, muestra), card_power(c_b, muestra)
            else:
                st_b = agent_b.encode_state(cards_b, muestra, table_hist, trick_idx, truco_lvl, True) if hasattr(agent_b, 'encode_state') else None
                c_b = cards_b.pop(agent_b.select_card_action(cards_b, st_b))
                st_a = agent_a.encode_state(cards_a, muestra, table_hist + [c_b], trick_idx, truco_lvl, True) if hasattr(agent_a, 'encode_state') else None
                c_a = cards_a.pop(agent_a.select_card_action(cards_a, st_a))
                table_hist.extend([c_b, c_a])
                pa, pb = card_power(c_a, muestra), card_power(c_b, muestra)

            if pa > pb:
                trick_wins.append(0)
                leader = 0
            elif pb > pa:
                trick_wins.append(1)
                leader = 1
            else:
                trick_wins.append(-1)
                leader = 0 if a_is_hand else 1

            val = 2 if truco_lvl >= 1 else 1
            if trick_wins.count(0) == 2 or (len(trick_wins) == 2 and trick_wins[0] == -1 and trick_wins[1] == 0):
                pts_a += val
                break
            elif trick_wins.count(1) == 2 or (len(trick_wins) == 2 and trick_wins[0] == -1 and trick_wins[1] == 1):
                pts_b += val
                break

        if pts_a > pts_b:
            wins_a += 1
        elif pts_b > pts_a:
            wins_b += 1
        else:
            ties += 1

    decided = max(1, wins_a + wins_b)
    win_rate = (wins_a / decided) * 100.0
    return wins_a, wins_b, ties, win_rate

def run_benchmark(hands_per_match: int = 1000):
    print("=" * 65)
    print("      BENCHMARK CIENTÍFICO DE INTELIGENCIA NEURONAL")
    print("  Conectoma FlyEM Male CNS v1.0 (Test de Ablación)")
    print("=" * 65)

    trained_fly = FlyBrainTrucoAgent()
    trained_fly.load_model("data/fly_truco_model.npz")

    naive_fly = NaiveFlyAgent()
    random_bot = RandomAgent()

    print(f"\nEjecutando {hands_per_match} manos por duelo...\n")

    # Duelo 1: Mosca Entrenada vs Jugador Aleatorio
    print(">>> Duelo 1: Mosca Entrenada (AlphaFly) vs Jugador Aleatorio...")
    w1, l1, t1, wr1 = simulate_duel(trained_fly, random_bot, hands_per_match)
    print(f"    Victorias: Mosca {w1} | Derrotas: {l1} | Empates: {t1}")
    print(f"    Tasa efectiva de victorias: {wr1:.1f}%\n")

    # Duelo 2: Mosca Entrenada vs Mosca Ingenua
    print(">>> Duelo 2 (Test de Ablación): Mosca Entrenada vs Mosca Ingenua (Sin Entrenar)...")
    w2, l2, t2, wr2 = simulate_duel(trained_fly, naive_fly, hands_per_match)
    print(f"    Victorias: Mosca {w2} | Derrotas: {l2} | Empates: {t2}")
    print(f"    Tasa efectiva de victorias: {wr2:.1f}%\n")

    plt.figure(figsize=(10, 5))
    categories = ['vs Jugador Aleatorio', 'vs Mosca Ingenua (Sin Entrenar)']
    win_rates = [wr1, wr2]
    colors = ['#2ca02c', '#1f77b4']

    bars = plt.bar(categories, win_rates, color=colors, width=0.45, edgecolor='black')
    plt.axhline(50, color='red', linestyle='--', label='Línea de Azar / Paridad (50%)')
    plt.title(f'Rendimiento del Cerebro de la Mosca (AlphaFly - {hands_per_match} manos por duelo)')
    plt.ylabel('% Victorias en manos decididas')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    chart_path = "data/fly_intelligence_benchmark.png"
    plt.savefig(chart_path, dpi=180)
    plt.close()

    print("=" * 65)
    print(" INFORME DE INTELIGENCIA BIOLÓGICA:")
    print(f" 1. Dominancia sobre el azar: {wr1:.1f}% de victorias frente al jugador aleatorio.")
    print(f" 2. Ventaja por plasticidad dopaminérgica: {wr2:.1f}% frente a la red virgen sin entrenar.")
    print(f" 3. Gráfico comparativo guardado en: {chart_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark(1000)
