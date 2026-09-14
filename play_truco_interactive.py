"""
Juego Interactivo de Truco Uruguayo: Humano vs Cerebro de la Mosca (FlyEM Connectome).
Incluye TELEMETRÍA NEURONAL BIOLÓGICA EN VIVO:
- Monitoreo en tiempo real de las 500 Células de Kenyon (KC)
- Disparo de las neuronas de salida (MBON)
- Liberación de Dopamina (PAM vs PPL1)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import random
import time
from typing import List, Optional, Tuple
import numpy as np

from truco_engine import (
    Card, create_deck, card_power, calculate_envido, 
    get_effective_piezas, has_flor, calculate_flor_points,
    card_label, TRUCO_VALUES, TRUCO_NAMES, TRUCO_REJECT_POINTS
)
from fly_brain_agent import FlyBrainTrucoAgent

def print_telemetry(agent: FlyBrainTrucoAgent, state_vec, action_name: str, dopamine_delta: float = 0.0):
    """Muestra la telemetría neuronal biológica de la mosca en vivo."""
    kc_act, mbon_act = agent.forward(state_vec)
    active_kc = (kc_act > 0).sum()
    pct_kc = (active_kc / len(kc_act)) * 100.0

    # Nivel de confianza / activación
    prob_truco = 1.0 / (1.0 + np.exp(-(mbon_act[3] - mbon_act[4]) * 2.0))
    prob_accept = 1.0 / (1.0 + np.exp(-(mbon_act[5] - mbon_act[6]) * 2.0))
    prob_envido = 1.0 / (1.0 + np.exp(-(mbon_act[7] - mbon_act[8]) * 2.0))
    prob_retruco = 1.0 / (1.0 + np.exp(-(mbon_act[9] - mbon_act[10]) * 2.0))
    prob_vale4 = 1.0 / (1.0 + np.exp(-(mbon_act[11] - mbon_act[12]) * 2.0))

    bar_len = 10
    fill = int(prob_truco * bar_len)
    truco_bar = "█" * fill + "░" * (bar_len - fill)

    print("  " + "═" * 58)
    print(f"  🧠 [TELEMETRÍA NEURONAL] Acción: {action_name}")
    print(f"  🔬 Células de Kenyon activas: {active_kc}/{len(kc_act)} ({pct_kc:.1f}% Sparse Firing)")
    print(f"  ⚡ Impulso Truco: [{truco_bar}] {prob_truco*100:.0f}% | Aceptación: {prob_accept*100:.0f}%")
    print(f"  🔥 Re-truco: {prob_retruco*100:.0f}% | Vale 4: {prob_vale4*100:.0f}% | Envido: {prob_envido*100:.0f}%")
    if dopamine_delta > 0:
        print(f"  🍬 Dopamina PAM: +{dopamine_delta:.1f} (Recompensa biológica)")
    elif dopamine_delta < 0:
        print(f"  ⚡ Dopamina PPL1: {dopamine_delta:.1f} (Aversión / Castigo)")
    print("  " + "═" * 58)

def play_match():
    print("\n" + "=" * 65)
    print("      TRUCO URUGUAYO: HUMANO vs EL CEREBRO DE LA MOSCA")
    print("  Conectoma FlyEM Male CNS v1.0 (AlphaFly & Telemetría en Vivo)")
    print("  Reglamento Oficial: Muestra, Piezas, Flor, Envido y Truco")
    print("=" * 65)

    fly = FlyBrainTrucoAgent()
    model_path = "data/fly_truco_model.npz"
    if os.path.exists(model_path):
        fly.load_model(model_path)
    else:
        print("Aviso: Modelo no entrenado aún, usando pesos biológicos directos de FlyEM.")

    human_score = 0
    fly_score = 0
    target_score = 15
    hand_num = 1

    print(f"\nPartida a {target_score} puntos de Truco Uruguayo.")
    print("¡Que comience el duelo!\n")

    while human_score < target_score and fly_score < target_score:
        print(f"\n{'#'*65}")
        print(f" MANO #{hand_num} | MARCADOR: Tú {human_score} - {fly_score} Mosca")
        print(f"{'#'*65}")

        deck = create_deck()
        random.shuffle(deck)
        muestra = deck.pop()

        human_cards = [deck.pop() for _ in range(3)]
        fly_cards = [deck.pop() for _ in range(3)]
        human_is_hand = (hand_num % 2 == 1)

        print(f"\n>>> LA MUESTRA ES: {card_label(muestra, muestra)}")
        piezas = get_effective_piezas(muestra)
        print(f"    (Piezas del palo {muestra.suit}: {piezas[0]}, {piezas[1]}, {piezas[2]}, {piezas[3]}, {piezas[4]})\n")

        print("Tus cartas:")
        for idx, c in enumerate(human_cards, 1):
            print(f"  ({idx}) {card_label(c, muestra)}")

        # 1. FLOR
        human_has_flor = has_flor(human_cards, muestra)
        fly_has_flor = has_flor(fly_cards, muestra)
        envido_canceled = False

        if human_has_flor and fly_has_flor:
            h_flor_pts = calculate_flor_points(human_cards, muestra)
            f_flor_pts = calculate_flor_points(fly_cards, muestra)
            print("\n🌸 ¡¡TIENES FLOR!!")
            time.sleep(0.4)
            print("🌸 ¡¡LA MOSCA TAMBIÉN TIENE FLOR!!")
            print(f"Tus tantos de Flor: {h_flor_pts} | Mosca canta: {f_flor_pts}")
            if h_flor_pts >= f_flor_pts:
                print(">> ¡Ganaste el lance de Flor! (+3 puntos)")
                human_score += 3
                fly.apply_dopamine_reward(-1.0)
            else:
                print(">> La mosca gana el lance de Flor. (+3 puntos)")
                fly_score += 3
                fly.apply_dopamine_reward(1.5)
            envido_canceled = True
        elif human_has_flor:
            h_flor_pts = calculate_flor_points(human_cards, muestra)
            print(f"\n🌸 ¡¡TIENES FLOR!! ({h_flor_pts} tantos) -> ¡Sumas 3 puntos!")
            human_score += 3
            print(">> La Flor anula el Envido. Pasamos directo al Truco.\n")
            envido_canceled = True
        elif fly_has_flor:
            f_flor_pts = calculate_flor_points(fly_cards, muestra)
            time.sleep(0.4)
            print(f"\n🌸 La mosca zumba y anuncia: ¡¡FLOR!! ({f_flor_pts} tantos) -> Suma 3 puntos.")
            fly_score += 3
            fly.apply_dopamine_reward(1.5)
            print(">> La Flor anula el Envido. Pasamos directo al Truco.\n")
            envido_canceled = True

        # 2. ENVIDO (si no hay Flor)
        if not envido_canceled:
            human_envido = calculate_envido(human_cards, muestra)
            fly_envido = calculate_envido(fly_cards, muestra)
            print(f"\n-> Tus puntos de Envido: {human_envido}")

            st_fly = fly.encode_state(fly_cards, muestra, [], 0, 0, False)

            if human_is_hand:
                resp = input("¿Cantas Envido? [s/n]: ").strip().lower()
                if resp == 's':
                    print("\nTú dices: ¡¡ENVIDO!!")
                    time.sleep(0.4)
                    fly_accepts = fly.decide_envido(st_fly)
                    print_telemetry(fly, st_fly, "Respuesta a Envido")
                    if fly_accepts:
                        print("La mosca responde: ¡QUIERO!")
                        print(f"Tus puntos: {human_envido} | Mosca canta: {fly_envido}")
                        if human_envido >= fly_envido:
                            print(">> ¡Ganaste el Envido! (+2 puntos)")
                            human_score += 2
                            fly.apply_dopamine_reward(-1.0)
                        else:
                            print(">> La mosca gana el Envido. (+2 puntos)")
                            fly_score += 2
                            fly.apply_dopamine_reward(1.5)
                    else:
                        print("La mosca responde: NO QUIERO (Se achica).")
                        print(">> Ganaste 1 punto de Envido.")
                        human_score += 1
            else:
                if fly.decide_envido(st_fly):
                    print_telemetry(fly, st_fly, "Canto de Envido")
                    print("\nLa mosca te mira fijo y canta: ¡¡ENVIDO!!")
                    resp = input("¿Aceptas el Envido? [1: Quiero, 2: No quiero]: ").strip()
                    if resp == '1':
                        print(f"Tus puntos: {human_envido} | Mosca canta: {fly_envido}")
                        if human_envido > fly_envido:
                            print(">> ¡Ganaste el Envido! (+2 puntos)")
                            human_score += 2
                            fly.apply_dopamine_reward(-1.0)
                        else:
                            print(">> La mosca gana el Envido. (+2 puntos)")
                            fly_score += 2
                            fly.apply_dopamine_reward(1.5)
                    else:
                        print("Tú dices: No quiero.")
                        print(">> La mosca se lleva 1 punto.")
                        fly_score += 1
                        fly.apply_dopamine_reward(0.8)

        # 3. TRUCO, RE-TRUCO Y VALE 4
        truco_level = 0
        canto_holder = None  # 0: Humano, 1: Mosca (quién tiene derecho a cantar el próximo nivel)
        trick_history = []
        table_played = []
        leader = 0 if human_is_hand else 1
        hand_over = False

        def fly_eval_truco_raise(c_level: int) -> Optional[int]:
            """Verifica si la mosca decide subir de nivel."""
            st_check = fly.encode_state(fly_cards, muestra, table_played, trick_idx, c_level, True)
            if c_level == 0 and fly.decide_truco(st_check):
                return 1
            elif c_level == 1 and fly.decide_retruco(st_check):
                return 2
            elif c_level == 2 and fly.decide_vale_cuatro(st_check):
                return 3
            return None

        def resolve_fly_call(target_level: int) -> Tuple[int, int, bool, int, Optional[int]]:
            """La mosca canta target_level (1: Truco, 2: Re-truco, 3: Vale 4)."""
            names = {1: "¡¡TRUCO!!", 2: "¡¡RE-TRUCO!!", 3: "¡¡VALE CUATRO!!"}
            pts_val = TRUCO_VALUES[target_level]
            prev_val = TRUCO_REJECT_POINTS.get(target_level, 1)

            print(f"\n🐝 La mosca hace zumbar sus alas y canta con fuerza: {names[target_level]} (Vale {pts_val} puntos)")
            if target_level == 1:
                ans = input("¿Qué respondes? [1: Quiero (2 pts), 2: No quiero / Mazo (1 pt), 3: ¡¡RE-TRUCO!! (3 pts)]: ").strip().lower()
                if ans == '3' or ans == 'r':
                    print("\n🔥 Tú respondes: ¡¡QUIERO RE-TRUCO!!")
                    time.sleep(0.4)
                    st_re = fly.encode_state(fly_cards, muestra, table_played, trick_idx, 2, True)
                    print_telemetry(fly, st_re, "Decisión de Aceptar Re-truco")
                    f_re = fly.decide_truco_response(st_re, 2)
                    if f_re == "no_quiero":
                        print("La mosca se achica ante tu Re-truco y se va al mazo. ¡Sumas 1 punto!")
                        fly.apply_dopamine_reward(-1.0)
                        return 1, 0, True, 2, None
                    elif f_re == "vale_4":
                        print("🐝 ¡La mosca no se achica y redobla: ¡¡QUIERO VALE CUATRO!! 💥")
                        ans_v4 = input("¿Aceptas el Vale 4? [1: Quiero (4 pts), 2: No quiero (mosca gana 2 pts)]: ").strip()
                        if ans_v4 == '1':
                            print("Tú dices: ¡¡QUIERO EL VALE CUATRO!!")
                            return 0, 0, False, 3, None
                        else:
                            print("Te fuiste al mazo. La mosca se lleva 2 puntos.")
                            fly.apply_dopamine_reward(2.0)
                            return 0, 2, True, 3, None
                    else:
                        print("La mosca responde: ¡¡QUIERO EL RE-TRUCO!!")
                        return 0, 0, False, 2, 1
                elif ans == '1' or ans == 'q':
                    print("Tú respondes: ¡Quiero!")
                    return 0, 0, False, 1, 0  # Humano tiene derecho a Re-truco
                else:
                    print(f"Te fuiste al mazo. La mosca suma {prev_val} punto(s).")
                    fly.apply_dopamine_reward(1.0)
                    return 0, prev_val, True, 1, None

            elif target_level == 2:
                ans = input("¿Qué respondes al Re-truco? [1: Quiero (3 pts), 2: No quiero (1 pt), 3: ¡¡VALE 4!! (4 pts)]: ").strip().lower()
                if ans == '3' or ans == 'v':
                    print("\n💥 Tú golpeas la mesa: ¡¡QUIERO VALE CUATRO!!")
                    time.sleep(0.4)
                    st_v4 = fly.encode_state(fly_cards, muestra, table_played, trick_idx, 3, True)
                    print_telemetry(fly, st_v4, "Decisión de Aceptar Vale 4")
                    f_v4 = fly.decide_truco_response(st_v4, 3)
                    if f_v4 == "no_quiero":
                        print("La mosca no quiso el Vale 4. ¡Sumas 2 puntos!")
                        fly.apply_dopamine_reward(-1.5)
                        return 2, 0, True, 3, None
                    else:
                        print("🐝 ¡La mosca zumba desafiante: ¡¡QUIERO EL VALE CUATRO!!")
                        return 0, 0, False, 3, None
                elif ans == '1' or ans == 'q':
                    print("Tú respondes: ¡Quiero!")
                    return 0, 0, False, 2, 0  # Humano tiene derecho a Vale 4
                else:
                    print(f"Te fuiste al mazo. La mosca suma {prev_val} puntos.")
                    fly.apply_dopamine_reward(1.5)
                    return 0, prev_val, True, 2, None

            elif target_level == 3:
                ans = input("¿Aceptas el Vale 4? [1: Quiero (4 pts), 2: No quiero (2 pts)]: ").strip()
                if ans == '1':
                    print("Tú dices: ¡¡QUIERO EL VALE 4!!")
                    return 0, 0, False, 3, None
                else:
                    print(f"Te fuiste al mazo. La mosca suma {prev_val} puntos.")
                    fly.apply_dopamine_reward(2.0)
                    return 0, prev_val, True, 3, None

            return 0, 0, False, target_level, None

        def resolve_human_call(target_level: int) -> Tuple[int, int, bool, int, Optional[int]]:
            """El humano canta target_level (1: Truco, 2: Re-truco, 3: Vale 4)."""
            names = {1: "¡¡TRUCO!!", 2: "¡¡RE-TRUCO!!", 3: "¡¡VALE CUATRO!!"}
            prev_val = TRUCO_REJECT_POINTS.get(target_level, 1)
            print(f"\n✋ Tú cantas: {names[target_level]} (Vale {TRUCO_VALUES[target_level]} puntos)")
            time.sleep(0.4)

            st_resp = fly.encode_state(fly_cards, muestra, table_played, trick_idx, target_level, True)
            print_telemetry(fly, st_resp, f"Decisión ante {names[target_level]}")
            f_resp = fly.decide_truco_response(st_resp, target_level)

            if f_resp == "no_quiero":
                print(f"La mosca se va al mazo ante tu canto. ¡Te llevas {prev_val} punto(s)!")
                human_pts = prev_val
                fly.apply_dopamine_reward(-0.8 * prev_val)
                return human_pts, 0, True, target_level, None

            elif f_resp == "retruco":
                print("🐝 ¡La mosca zumba y retruca: ¡¡QUIERO RE-TRUCO!! 🔥")
                ans = input("¿Qué respondes al Re-truco? [1: Quiero (3 pts), 2: No quiero (mosca gana 1 pt), 3: ¡¡VALE 4!! (4 pts)]: ").strip().lower()
                if ans == '3' or ans == 'v':
                    print("\n💥 Tú dices: ¡¡QUIERO VALE CUATRO!!")
                    time.sleep(0.4)
                    st_v4 = fly.encode_state(fly_cards, muestra, table_played, trick_idx, 3, True)
                    print_telemetry(fly, st_v4, "Decisión de Aceptar Vale 4")
                    f_v4 = fly.decide_truco_response(st_v4, 3)
                    if f_v4 == "no_quiero":
                        print("La mosca no quiso el Vale 4. ¡Te llevas 2 puntos!")
                        fly.apply_dopamine_reward(-1.5)
                        return 2, 0, True, 3, None
                    else:
                        print("🐝 La mosca responde: ¡¡QUIERO EL VALE CUATRO!!")
                        return 0, 0, False, 3, None
                elif ans == '1' or ans == 'q':
                    print("Tú dices: ¡Quiero el Re-truco!")
                    return 0, 0, False, 2, 0
                else:
                    print("Te fuiste al mazo. La mosca se lleva 1 punto.")
                    fly.apply_dopamine_reward(1.0)
                    return 0, 1, True, 2, None

            elif f_resp == "vale_4":
                print("🐝 ¡La mosca desafía a muerte: ¡¡QUIERO VALE CUATRO!! 💥")
                ans = input("¿Qué respondes? [1: Quiero (4 pts), 2: No quiero (mosca gana 2 pts)]: ").strip()
                if ans == '1':
                    print("Tú dices: ¡¡QUIERO EL VALE 4!!")
                    return 0, 0, False, 3, None
                else:
                    print("Te fuiste al mazo. La mosca se lleva 2 puntos.")
                    fly.apply_dopamine_reward(2.0)
                    return 0, 2, True, 3, None

            else:
                print(f"🐝 La mosca responde: ¡¡QUIERO EL {names[target_level].replace('¡', '').replace('!', '')}!!")
                return 0, 0, False, target_level, 1  # Mosca tiene derecho a subir

        for trick_idx in range(3):
            if hand_over or not human_cards or not fly_cards:
                break

            print(f"\n--- BAZA #{trick_idx + 1} --- (Nivel actual de apuesta: {TRUCO_NAMES[truco_level]})")

            if leader == 0:
                print("Tus cartas disponibles:")
                for i, c in enumerate(human_cards, 1):
                    print(f"  ({i}) {card_label(c, muestra)}")

                prompt_actions = ["1, 2, 3: tirar carta"]
                if truco_level == 0:
                    prompt_actions.append("[T]ruco (2 pts)")
                elif truco_level == 1 and canto_holder == 0:
                    prompt_actions.append("[R]e-truco (3 pts)")
                elif truco_level == 2 and canto_holder == 0:
                    prompt_actions.append("[V]ale 4 (4 pts)")
                prompt_actions.append("[M]azo")

                choice = input(f"Elige ({', '.join(prompt_actions)}): ").strip().lower()

                if choice == 'm':
                    p_loss = TRUCO_VALUES.get(truco_level, 1)
                    print(f"Te fuiste al mazo. La mosca suma {p_loss} punto(s).")
                    fly_score += p_loss
                    fly.apply_dopamine_reward(1.0)
                    hand_over = True
                    break

                if choice == 't' and truco_level == 0:
                    h_add, f_add, over, truco_level, canto_holder = resolve_human_call(1)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break
                    choice = input("Ahora tira tu carta (1, 2, 3): ").strip()
                elif choice == 'r' and truco_level == 1 and canto_holder == 0:
                    h_add, f_add, over, truco_level, canto_holder = resolve_human_call(2)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break
                    choice = input("Ahora tira tu carta (1, 2, 3): ").strip()
                elif choice == 'v' and truco_level == 2 and canto_holder == 0:
                    h_add, f_add, over, truco_level, canto_holder = resolve_human_call(3)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break
                    choice = input("Ahora tira tu carta (1, 2, 3): ").strip()

                try:
                    c_idx = int(choice) - 1
                    if c_idx < 0 or c_idx >= len(human_cards):
                        c_idx = 0
                except ValueError:
                    c_idx = 0

                h_card = human_cards.pop(c_idx)
                print(f"Tiraste: {card_label(h_card, muestra)}")

                # Turno Mosca
                time.sleep(0.4)
                st_card = fly.encode_state(fly_cards, muestra, table_played + [h_card], trick_idx, truco_level, True)

                # Mosca puede subir si tiene el canto o si truco_level == 0
                f_call = fly_eval_truco_raise(truco_level)
                if f_call is not None and (truco_level == 0 or canto_holder == 1):
                    h_add, f_add, over, truco_level, canto_holder = resolve_fly_call(f_call)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break

                fly_idx = fly.select_card_action(fly_cards, st_card)
                f_card = fly_cards.pop(fly_idx)
                print_telemetry(fly, st_card, f"Jugada de Carta #{fly_idx+1}")
                print(f"La mosca juega: {card_label(f_card, muestra)}")

                pow_h = card_power(h_card, muestra)
                pow_f = card_power(f_card, muestra)

            else:
                # Lidera Mosca
                time.sleep(0.4)
                st_card = fly.encode_state(fly_cards, muestra, table_played, trick_idx, truco_level, True)

                # Mosca puede subir si tiene el canto o si truco_level == 0
                f_call = fly_eval_truco_raise(truco_level)
                if f_call is not None and (truco_level == 0 or canto_holder == 1):
                    h_add, f_add, over, truco_level, canto_holder = resolve_fly_call(f_call)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break

                fly_idx = fly.select_card_action(fly_cards, st_card)
                f_card = fly_cards.pop(fly_idx)
                print_telemetry(fly, st_card, f"Apertura con Carta #{fly_idx+1}")
                print(f"La mosca abre con: {card_label(f_card, muestra)}")

                print("\nTus cartas disponibles:")
                for i, c in enumerate(human_cards, 1):
                    print(f"  ({i}) {card_label(c, muestra)}")

                prompt_actions = ["1, 2, 3: tirar carta"]
                if truco_level == 0:
                    prompt_actions.append("[T]ruco (2 pts)")
                elif truco_level == 1 and canto_holder == 0:
                    prompt_actions.append("[R]e-truco (3 pts)")
                elif truco_level == 2 and canto_holder == 0:
                    prompt_actions.append("[V]ale 4 (4 pts)")
                prompt_actions.append("[M]azo")

                choice = input(f"Elige ({', '.join(prompt_actions)}): ").strip().lower()

                if choice == 'm':
                    p_loss = TRUCO_VALUES.get(truco_level, 1)
                    print(f"Te fuiste al mazo. La mosca suma {p_loss} punto(s).")
                    fly_score += p_loss
                    fly.apply_dopamine_reward(1.0)
                    hand_over = True
                    break

                if choice == 't' and truco_level == 0:
                    h_add, f_add, over, truco_level, canto_holder = resolve_human_call(1)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break
                    choice = input("Ahora tira tu carta (1, 2, 3): ").strip()
                elif choice == 'r' and truco_level == 1 and canto_holder == 0:
                    h_add, f_add, over, truco_level, canto_holder = resolve_human_call(2)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break
                    choice = input("Ahora tira tu carta (1, 2, 3): ").strip()
                elif choice == 'v' and truco_level == 2 and canto_holder == 0:
                    h_add, f_add, over, truco_level, canto_holder = resolve_human_call(3)
                    human_score += h_add
                    fly_score += f_add
                    if over:
                        hand_over = True
                        break
                    choice = input("Ahora tira tu carta (1, 2, 3): ").strip()

                try:
                    c_idx = int(choice) - 1
                    if c_idx < 0 or c_idx >= len(human_cards):
                        c_idx = 0
                except ValueError:
                    c_idx = 0

                h_card = human_cards.pop(c_idx)
                print(f"Tiraste: {card_label(h_card, muestra)}")

                pow_h = card_power(h_card, muestra)
                pow_f = card_power(f_card, muestra)

            table_played.extend([h_card, f_card])

            # Ganador de baza
            if pow_h > pow_f:
                print(">> Ganaste la baza.")
                trick_history.append(0)
                leader = 0
            elif pow_f > pow_h:
                print(">> La mosca mató tu carta y gana la baza.")
                trick_history.append(1)
                leader = 1
            else:
                print(">> ¡Parda! (Empate en la baza).")
                trick_history.append(-1)
                leader = 0 if human_is_hand else 1

            val_pts = TRUCO_VALUES[truco_level]
            dop_mult = 1.0 + 0.5 * truco_level

            if trick_history.count(0) == 2:
                print(f"\n>>> ¡¡GANASTE LA MANO DE TRUCO!! (+{val_pts} puntos)")
                human_score += val_pts
                fly.apply_dopamine_reward(-1.5 * dop_mult)
                hand_over = True
                break
            elif trick_history.count(1) == 2:
                print(f"\n>>> La mosca te ganó la mano de Truco. (+{val_pts} puntos para la mosca)")
                fly_score += val_pts
                fly.apply_dopamine_reward(2.0 * dop_mult)
                hand_over = True
                break
            elif len(trick_history) == 2 and trick_history[0] == -1:
                if trick_history[1] == 0:
                    print(f"\n>>> Desempardaste en segunda: ¡Ganaste el Truco! (+{val_pts} puntos)")
                    human_score += val_pts
                    fly.apply_dopamine_reward(-1.5 * dop_mult)
                    hand_over = True
                    break
                elif trick_history[1] == 1:
                    print(f"\n>>> La mosca desempardó en segunda: Gana el Truco. (+{val_pts} puntos)")
                    fly_score += val_pts
                    fly.apply_dopamine_reward(2.0 * dop_mult)
                    hand_over = True
                    break

        if not hand_over and len(trick_history) == 3:
            val_pts = TRUCO_VALUES[truco_level]
            dop_mult = 1.0 + 0.5 * truco_level
            if trick_history.count(0) > trick_history.count(1):
                print(f"\n>>> ¡Ganaste en tercera mano! (+{val_pts} puntos)")
                human_score += val_pts
                fly.apply_dopamine_reward(-1.5 * dop_mult)
            elif trick_history.count(1) > trick_history.count(0):
                print(f"\n>>> La mosca te ganó en tercera mano. (+{val_pts} puntos)")
                fly_score += val_pts
                fly.apply_dopamine_reward(2.0 * dop_mult)
            else:
                print(f"\n>>> Triple parda. Gana la mano: {'Tú' if human_is_hand else 'La mosca'}")
                if human_is_hand:
                    human_score += val_pts
                    fly.apply_dopamine_reward(-1.0 * dop_mult)
                else:
                    fly_score += val_pts
                    fly.apply_dopamine_reward(1.5 * dop_mult)

        hand_num += 1

    print("\n" + "=" * 65)
    if human_score >= target_score:
        print(" ¡FELICITACIONES! DERROTASTE AL CEREBRO DE LA MOSCA EN EL TRUCO URUGUAYO.")
    else:
        print(" EL CEREBRO DE LA MOSCA TE HA GANADO LA PARTIDA. ¡UN MAESTRO DEL TRUCO!")
    print(f" Marcador final: Humano {human_score} - {fly_score} Mosca")
    print("=" * 65)

if __name__ == "__main__":
    play_match()
