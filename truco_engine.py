"""
Motor de Truco Uruguayo Oficial con Muestra, Piezas, Flor, Envido y Truco.
Implementa las reglas tradicionales uruguayas completas.
"""

import random
from typing import List, Tuple, Optional, Dict

SUITS = ["Espada", "Basto", "Oro", "Copa"]
NUMBERS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
PIEZA_NUMBERS = [2, 4, 5, 11, 10]

class Card:
    def __init__(self, number: int, suit: str):
        self.number = number
        self.suit = suit

    def __repr__(self):
        return f"{self.number} de {self.suit}"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.number == other.number and self.suit == other.suit

    def __hash__(self):
        return hash((self.number, self.suit))

def create_deck() -> List[Card]:
    """Crea una baraja española de 40 cartas."""
    return [Card(n, s) for s in SUITS for n in NUMBERS]

def get_effective_piezas(muestra: Card) -> List[int]:
    """
    En el Truco Uruguayo, las piezas son 2, 4, 5, 11, 10 del palo de la muestra.
    Si la muestra es una de las piezas (ej. el 2, 4, 5, 11 o 10), el Rey (12) toma el lugar de esa pieza.
    """
    piezas = [2, 4, 5, 11, 10]
    if muestra.number in piezas:
        idx = piezas.index(muestra.number)
        piezas[idx] = 12
    return piezas

def card_power(card: Card, muestra: Card) -> int:
    """
    Calcula el poder de combate de una carta según la jerarquía del Truco Uruguayo.
    Mayor valor = mayor poder.
    """
    piezas = get_effective_piezas(muestra)

    # 1. ¿Es Pieza? (Mismo palo que la muestra y en la lista de piezas efectivas)
    if card.suit == muestra.suit and card.number in piezas:
        idx = piezas.index(card.number)
        # 2=100, 4=99, 5=98, 11=97, 10=96 (o 12 si reemplazó)
        return 100 - idx

    # 2. ¿Es Mata común? (Solo si no fue pieza del palo de la muestra)
    if card.number == 1 and card.suit == "Espada":
        return 90  # Macho
    if card.number == 1 and card.suit == "Basto":
        return 89  # Hembra
    if card.number == 7 and card.suit == "Espada":
        return 88
    if card.number == 7 and card.suit == "Oro":
        return 87

    # 3. Cartas Comunes
    # 3s
    if card.number == 3:
        return 80
    # 2s comunes
    if card.number == 2:
        return 75
    # 1s comunes (Copa, Oro)
    if card.number == 1:
        return 70
    # 12s comunes
    if card.number == 12:
        return 65
    # 11s comunes
    if card.number == 11:
        return 60
    # 10s comunes
    if card.number == 10:
        return 55
    # 7s comunes (Copa, Basto)
    if card.number == 7:
        return 50
    # 6s
    if card.number == 6:
        return 40
    # 5s comunes
    if card.number == 5:
        return 30
    # 4s comunes
    if card.number == 4:
        return 20

    return 10

def card_envido_val(card: Card) -> int:
    """Valor para envido común: 1 al 7 valen su índice; figuras 10, 11, 12 valen cero."""
    if card.number in [10, 11, 12]:
        return 0
    return card.number

def has_flor(hand: List[Card], muestra: Card) -> bool:
    """
    Reglamento oficial de Truco Uruguayo:
    Se dice que un jugador tiene Flor cuando tiene:
    - Dos o tres piezas.
    - O una pieza y dos cartas de un mismo palo.
    - O tres cartas de un mismo palo.
    """
    piezas = get_effective_piezas(muestra)
    piezas_in_hand = [c for c in hand if c.suit == muestra.suit and c.number in piezas]
    comunes_in_hand = [c for c in hand if c not in piezas_in_hand]

    # Condición 1: Dos o tres piezas
    if len(piezas_in_hand) >= 2:
        return True

    # Condición 2: Una pieza y dos cartas del mismo palo entre sí
    if len(piezas_in_hand) == 1:
        if len(comunes_in_hand) == 2 and comunes_in_hand[0].suit == comunes_in_hand[1].suit:
            return True

    # Condición 3: Tres cartas del mismo palo (sin piezas)
    if len(piezas_in_hand) == 0:
        if len(hand) == 3 and hand[0].suit == hand[1].suit == hand[2].suit:
            return True

    return False

def calculate_flor_points(hand: List[Card], muestra: Card) -> int:
    """
    Valoración oficial de Flor (para el caso de 'Con flor envido'):
    - Se contabiliza un valor mínimo base de 20.
    - Las piezas suman su valor de envido restando 20:
        2 vale 10, 4 vale 9, 5 vale 8, 11 vale 7, 10 vale 7.
    - El resto de cartas suman según su índice (1..7, figuras valen 0).
    """
    piezas = get_effective_piezas(muestra)
    pieza_flor_val = {piezas[0]: 10, piezas[1]: 9, piezas[2]: 8, piezas[3]: 7, piezas[4]: 7}

    total = 20
    for c in hand:
        if c.suit == muestra.suit and c.number in piezas:
            total += pieza_flor_val[c.number]
        else:
            total += card_envido_val(c)
    return total

def calculate_envido(hand: List[Card], muestra: Card) -> int:
    """
    Cálculo oficial de Envido del Truco Uruguayo (se disputa solo si no hay Flor):
    - En caso de tener una pieza: se suma el valor de la pieza (30, 29, 28, 27, 27)
      y el índice superior de las otras dos cartas (figuras valen 0).
    - En caso de no tener piezas y tener 2 cartas de un mismo palo:
      se suman los índices de ambas y se añaden 20 más.
    - Si las 3 cartas son de diferentes palos:
      el valor del envido es el índice más alto de las 3 cartas.
    """
    piezas = get_effective_piezas(muestra)
    pieza_values = {piezas[0]: 30, piezas[1]: 29, piezas[2]: 28, piezas[3]: 27, piezas[4]: 27}

    piezas_in_hand = [c for c in hand if c.suit == muestra.suit and c.number in piezas]
    comunes_in_hand = [c for c in hand if c not in piezas_in_hand]

    # 1. Con una pieza (Nota: si tuviera 2 piezas, es Flor y no se juega envido)
    if len(piezas_in_hand) >= 1:
        p_val = pieza_values[piezas_in_hand[0].number]
        # El índice superior de las otras cartas
        max_comun = max([card_envido_val(c) for c in comunes_in_hand]) if comunes_in_hand else 0
        return p_val + max_comun

    # 2. Sin piezas: Agrupar por palo
    by_suit: Dict[str, List[int]] = {}
    for c in hand:
        by_suit.setdefault(c.suit, []).append(card_envido_val(c))

    max_pts = 0
    for s, vals in by_suit.items():
        if len(vals) >= 2:
            s_vals = sorted(vals, reverse=True)
            pts = 20 + s_vals[0] + s_vals[1]
            if pts > max_pts:
                max_pts = pts
        else:
            if vals[0] > max_pts:
                max_pts = vals[0]

    return max_pts

TRUCO_VALUES = {0: 1, 1: 2, 2: 3, 3: 4}
TRUCO_NAMES = {0: "Normal (1 pt)", 1: "Truco (2 pts)", 2: "Re-truco (3 pts)", 3: "Vale 4 (4 pts)"}
TRUCO_REJECT_POINTS = {1: 1, 2: 2, 3: 3}

def card_label(card: Card, muestra: Card) -> str:
    """Genera la etiqueta legible y descriptiva de una carta según las reglas de Truco Uruguayo."""
    piezas = get_effective_piezas(muestra)
    if card.suit == muestra.suit and card.number in piezas:
        p_name = {
            piezas[0]: "PIEZA MÁXIMA (2)",
            piezas[1]: "PIEZA (4)",
            piezas[2]: "PIEZA (5)",
            piezas[3]: "PIEZA (11)",
            piezas[4]: "PIEZA (10)"
        }.get(card.number, "PIEZA (12)")
        return f"[{card.number} de {card.suit} 🌟 {p_name}]"
    if card.number == 1 and card.suit == "Espada":
        return f"[{card.number} de {card.suit} ⚔️ MACHO]"
    if card.number == 1 and card.suit == "Basto":
        return f"[{card.number} de {card.suit} 🌿 HEMBRA]"
    if card.number == 7 and card.suit == "Espada":
        return f"[{card.number} de {card.suit} ⚡ 7 de Espada]"
    if card.number == 7 and card.suit == "Oro":
        return f"[{card.number} de {card.suit} 🪙 7 de Oro]"
    return f"[{card.number} de {card.suit}]"

class TrucoGame:
    """Gestiona una mano completa de Truco Uruguayo mano a mano."""
    def __init__(self, p0_is_hand: bool = True):
        self.deck = create_deck()
        random.shuffle(self.deck)

        self.muestra = self.deck.pop()
        self.hands = [
            [self.deck.pop() for _ in range(3)],
            [self.deck.pop() for _ in range(3)]
        ]
        self.hand_turn = 0 if p0_is_hand else 1

        self.scores = [0, 0]
        self.played_cards = [[], []]
        self.trick_winners = []

        self.envido_state = "NONE"
        self.truco_state = "NONE"
        self.truco_level = 0
        self.canto_owner = None # Quién tiene el derecho de cantar el próximo nivel de Truco

    def resolve_trick(self, card0: Card, card1: Card) -> int:
        p0 = card_power(card0, self.muestra)
        p1 = card_power(card1, self.muestra)
        if p0 > p1:
            return 0
        elif p1 > p0:
            return 1
        else:
            return -1 # Parda
