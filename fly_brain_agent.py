"""
Agente de Truco Uruguayo basado en el circuito real del Mushroom Body (Cuerpos Fúngicos)
del conectoma de FlyEM Male CNS v1.0.

Estructura biológica:
- Células de Kenyon (KC): Representación abstracta dispersa (Sparse Coding) del estado del juego.
- Neuronas de Salida del Mushroom Body (MBON): Decisión de acciones (qué carta jugar, cantar Truco/Envido, aceptar o irse al mazo).
- Neuronas Dopaminérgicas (PAM/PPL1): Modulación por refuerzo biológico (azúcar/recompensa al ganar, shock/aversión al perder).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from truco_engine import Card, card_power, calculate_envido, get_effective_piezas

DATA_DIR = Path(r"C:\Users\nahue\OneDrive\Escritorio\UCU\Proyectos\mosca\data\flat-connectome")
WEIGHTS_FILE = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
ANN_FILE = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CACHE_PATH = Path(r"C:\Users\nahue\OneDrive\Escritorio\UCU\Proyectos\mosca\data\mushroom_body_subcircuit.npz")

def extract_or_load_subcircuit(num_kc: int = 500, num_mbon: int = 24) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Carga o extrae la matriz de pesos biológicos reales KC -> MBON desde el conectoma de FlyEM.
    """
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        return data["weights"], data["kc_ids"].tolist(), data["mbon_ids"].tolist()

    print("Extrayendo subcircuito biológico KC -> MBON desde el dataset de FlyEM...")
    ann = pd.read_feather(ANN_FILE)
    w = pd.read_feather(WEIGHTS_FILE)

    kc_ids = ann[ann['type'].str.contains('KC', na=False)]['bodyId'].unique()
    mbon_ids = ann[ann['type'].str.contains('MBON', na=False)]['bodyId'].unique()

    sub_w = w[w['body_pre'].isin(kc_ids) & w['body_post'].isin(mbon_ids)]
    top_kc = sub_w['body_pre'].value_counts().head(num_kc).index.tolist()
    top_mbon = sub_w['body_post'].value_counts().head(num_mbon).index.tolist()

    kc_map = {b_id: i for i, b_id in enumerate(top_kc)}
    mbon_map = {b_id: i for i, b_id in enumerate(top_mbon)}

    W = np.zeros((num_kc, num_mbon), dtype=np.float32)
    filtered = sub_w[sub_w['body_pre'].isin(top_kc) & sub_w['body_post'].isin(top_mbon)]

    for _, row in filtered.iterrows():
        i = kc_map[row['body_pre']]
        j = mbon_map[row['body_post']]
        W[i, j] = float(row['weight'])

    # Normalizar escala de pesos biológicos iniciales
    if W.max() > 0:
        W = W / W.max() * 0.5

    # Guardar en caché para inicios instantáneos
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, weights=W, kc_ids=np.array(top_kc), mbon_ids=np.array(top_mbon))
    print(f"Subcircuito biológico extraído ({num_kc} KCs x {num_mbon} MBONs) y guardado en caché.")
    return W, top_kc, top_mbon

class FlyBrainTrucoAgent:
    def __init__(self, num_kc: int = 500, num_mbon: int = 24, learning_rate: float = 0.015):
        self.num_kc = num_kc
        self.num_mbon = num_mbon
        self.lr = learning_rate

        # 1. Matriz biológica inicial KC -> MBON
        self.W_bio, self.kc_ids, self.mbon_ids = extract_or_load_subcircuit(num_kc, num_mbon)
        # Copia de trabajo plástica modulada por dopamina
        self.W = self.W_bio.copy()

        # 2. Proyección Sensorial (Lóbulo Antenal / Neuronas de Proyección -> Células de Kenyon)
        # En la mosca real, cada KC recibe conexiones de ~6-8 PNs de forma dispersa
        self.input_dim = 16 # Vector de estado del juego de Truco
        np.random.seed(42)
        self.W_pn_kc = np.random.randn(self.input_dim, self.num_kc).astype(np.float32) * 0.5
        # Sparsity en la proyección de entrada
        mask = np.random.rand(self.input_dim, self.num_kc) < 0.2
        self.W_pn_kc *= mask

        # Trazas de elegibilidad para asignación temporal de crédito (Dopamina retroactiva)
        self.eligibility = np.zeros_like(self.W)
        self.dopamine_history = []

    def encode_state(self, hand: List[Card], muestra: Card, table_cards: List[Card],
                     trick_idx: int, truco_level: int, envido_called: bool) -> np.ndarray:
        """
        Codifica el estado del juego en un vector continuo para alimentar el cerebro:
        - Poder de las cartas en mano
        - Envido propio
        - Poder de cartas en la mesa
        - Nivel de apuesta actual
        """
        vec = np.zeros(self.input_dim, dtype=np.float32)

        # 1. Poder de las 3 cartas (normalizado 0..1)
        for i in range(3):
            if i < len(hand):
                vec[i] = card_power(hand[i], muestra) / 100.0
            else:
                vec[i] = 0.0

        # 2. Puntos de envido propios (normalizado 0..1 sobre 37 pts máximos reales)
        pts_env = calculate_envido(hand, muestra) if hand and muestra else 0
        vec[3] = float(np.clip(pts_env / 37.0, 0.0, 1.0))

        # 3. ¿La muestra es pieza?
        piezas = get_effective_piezas(muestra)
        vec[4] = 1.0 if muestra.number in piezas else 0.0

        # 4. Cartas en mesa
        for i, c in enumerate(table_cards[:4]):
            vec[5 + i] = card_power(c, muestra) / 100.0

        # 5. Mano actual (ronda 1, 2 o 3)
        vec[9 + trick_idx] = 1.0

        # 6. Apuestas (Truco / Envido)
        vec[12] = truco_level / 4.0 # 0=none, 1=truco, 2=retruco, 3=vale4
        vec[13] = 1.0 if envido_called else 0.0
        vec[14] = float(len(hand)) / 3.0 # Cartas restantes
        vec[15] = 1.0 # Bias constante

        return vec

    def forward(self, state_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación neuronal:
        Entrada -> Células de Kenyon (Sparse Firing, top 10%) -> MBONs
        """
        # Excitación en Kenyon Cells
        kc_potential = np.dot(state_vec, self.W_pn_kc)
        # En la mosca real, la inhibición por APL genera disparo disperso (Sparse Coding)
        # Activamos solo el top 10% de las neuronas con mayor potencial
        threshold = np.percentile(kc_potential, 90)
        kc_activity = np.maximum(0.0, kc_potential - threshold)
        if kc_activity.max() > 0:
            kc_activity /= kc_activity.max()

        # Señal hacia las neuronas de salida (MBONs)
        mbon_activity = np.dot(kc_activity, self.W)
        return kc_activity, mbon_activity

    def select_card_action(self, hand: List[Card], state_vec: np.ndarray, temperature: float = 0.5) -> int:
        """
        MBONs 0 a 2 controlan qué carta tirar de la mano disponible.
        """
        kc_act, mbon_act = self.forward(state_vec)
        # Tomar los primeros 3 MBONs correspondientes a las cartas
        card_scores = mbon_act[:len(hand)]

        # Softmax con temperatura
        exp_s = np.exp(card_scores / max(0.1, temperature))
        probs = exp_s / np.sum(exp_s)
        chosen_idx = int(np.random.choice(len(hand), p=probs))

        # Registrar traza de elegibilidad
        # Pre-sináptico (KC) x Post-sináptico (MBON de la acción elegida)
        grad = np.zeros_like(self.W)
        grad[:, chosen_idx] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad

        return chosen_idx

    def decide_truco(self, state_vec: np.ndarray) -> bool:
        """MBON 3 y 4 deciden si cantar Truco inicial."""
        kc_act, mbon_act = self.forward(state_vec)
        # MBON 3: Cantar Truco, MBON 4: Mantener silencio
        diff = mbon_act[3] - mbon_act[4]
        prob_call = 1.0 / (1.0 + np.exp(-diff * 2.0))
        choice = bool(np.random.rand() < prob_call)
        grad = np.zeros_like(self.W)
        grad[:, 3 if choice else 4] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad
        return choice

    def decide_accept_truco(self, state_vec: np.ndarray) -> bool:
        """MBON 5 y 6 deciden si Querer Truco o irse al mazo."""
        kc_act, mbon_act = self.forward(state_vec)
        # MBON 5: Quiero, MBON 6: No Quiero (Mazo)
        diff = mbon_act[5] - mbon_act[6]
        prob_accept = 1.0 / (1.0 + np.exp(-diff * 2.0))
        choice = bool(np.random.rand() < prob_accept)
        grad = np.zeros_like(self.W)
        grad[:, 5 if choice else 6] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad
        return choice

    def decide_retruco(self, state_vec: np.ndarray) -> bool:
        """MBON 9 y 10 deciden si cantar / subir a Re-truco."""
        kc_act, mbon_act = self.forward(state_vec)
        diff = mbon_act[9] - mbon_act[10]
        prob_call = 1.0 / (1.0 + np.exp(-diff * 2.0))
        choice = bool(np.random.rand() < prob_call)
        grad = np.zeros_like(self.W)
        grad[:, 9 if choice else 10] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad
        return choice

    def decide_vale_cuatro(self, state_vec: np.ndarray) -> bool:
        """MBON 11 y 12 deciden si cantar / subir a Vale 4."""
        kc_act, mbon_act = self.forward(state_vec)
        diff = mbon_act[11] - mbon_act[12]
        prob_call = 1.0 / (1.0 + np.exp(-diff * 2.0))
        choice = bool(np.random.rand() < prob_call)
        grad = np.zeros_like(self.W)
        grad[:, 11 if choice else 12] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad
        return choice

    def decide_truco_response(self, state_vec: np.ndarray, current_level: int) -> str:
        """
        Determina la respuesta de la mosca ante un envite de Truco, Re-truco o Vale 4:
        - current_level == 1 (Truco): 'no_quiero', 'quiero', o 'retruco'.
        - current_level == 2 (Re-truco): 'no_quiero', 'quiero', o 'vale_4'.
        - current_level == 3 (Vale 4): 'no_quiero' o 'quiero'.
        """
        kc_act, mbon_act = self.forward(state_vec)

        # Decisión base de aceptación
        diff_acc = mbon_act[5] - mbon_act[6]
        prob_accept = 1.0 / (1.0 + np.exp(-diff_acc * 2.0))
        wants = bool(np.random.rand() < prob_accept)

        if not wants:
            grad = np.zeros_like(self.W)
            grad[:, 6] = kc_act
            self.eligibility = 0.9 * self.eligibility + grad
            return "no_quiero"

        # Si quiere, evalúa si redobla la apuesta (Raise)
        if current_level == 1:
            diff_re = mbon_act[9] - mbon_act[10]
            prob_re = 1.0 / (1.0 + np.exp(-diff_re * 2.0))
            if np.random.rand() < prob_re:
                grad = np.zeros_like(self.W)
                grad[:, 9] = kc_act
                self.eligibility = 0.9 * self.eligibility + grad
                return "retruco"
        elif current_level == 2:
            diff_v4 = mbon_act[11] - mbon_act[12]
            prob_v4 = 1.0 / (1.0 + np.exp(-diff_v4 * 2.0))
            if np.random.rand() < prob_v4:
                grad = np.zeros_like(self.W)
                grad[:, 11] = kc_act
                self.eligibility = 0.9 * self.eligibility + grad
                return "vale_4"

        grad = np.zeros_like(self.W)
        grad[:, 5] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad
        return "quiero"

    def decide_raise_truco(self, state_vec: np.ndarray, current_level: int) -> Optional[str]:
        """
        Evalúa si la mosca decide subir la apuesta en su turno cuando tiene el derecho al canto:
        - Si level == 1: puede cantar 'retruco'.
        - Si level == 2: puede cantar 'vale_4'.
        """
        if current_level == 1 and self.decide_retruco(state_vec):
            return "retruco"
        elif current_level == 2 and self.decide_vale_cuatro(state_vec):
            return "vale_4"
        return None

    def decide_envido(self, state_vec: np.ndarray) -> bool:
        """MBON 7 y 8 deciden si cantar Envido."""
        kc_act, mbon_act = self.forward(state_vec)
        diff = mbon_act[7] - mbon_act[8]

        # Modulación biológica por umbral de tantos (Sensory Gating):
        envido_norm = state_vec[3] if len(state_vec) > 3 else 0.5
        if envido_norm < (26.0 / 37.0):
            diff -= 6.0 # Freno absoluto a cantar con mano baja (<26)
        elif envido_norm >= (28.0 / 37.0):
            diff += 4.0 # Impulso a cantar con mano buena (>=28)

        prob_envido = 1.0 / (1.0 + np.exp(-diff * 2.0))
        choice = bool(np.random.rand() < prob_envido)
        grad = np.zeros_like(self.W)
        grad[:, 7 if choice else 8] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad
        return choice

    def decide_accept_envido(self, state_vec: np.ndarray) -> bool:
        """MBON 13 y 14 deciden si aceptar Envido (Quiero) o rechazar (No Quiero)."""
        kc_act, mbon_act = self.forward(state_vec)
        diff = mbon_act[13] - mbon_act[14]

        # Modulación biológica por umbral de tantos:
        envido_norm = state_vec[3] if len(state_vec) > 3 else 0.5
        if envido_norm < (24.0 / 37.0):
            # Mano baja/basura (<24 tantos): fuerte aversión / rechazo total
            diff -= 6.0
        elif envido_norm >= (28.0 / 37.0):
            # Mano alta (>=28 tantos): fuerte impulso apetitivo de aceptación
            diff += 3.5 + 2.5 * (envido_norm - 28.0 / 37.0)

        prob = 1.0 / (1.0 + np.exp(-diff * 2.0))
        choice = bool(np.random.rand() < prob)
        grad = np.zeros_like(self.W)
        grad[:, 13 if choice else 14] = kc_act
        self.eligibility = 0.9 * self.eligibility + grad
        return choice

    def get_detailed_telemetry(self, state_vec: np.ndarray) -> Dict:
        """
        Genera telemetría biológica detallada completa para el panel visual interactivo.
        """
        kc_act, mbon_act = self.forward(state_vec)
        active_kc = int((kc_act > 0).sum())
        pct_kc = float((active_kc / len(kc_act)) * 100.0)

        # Probabilidades / impulsos de acción MBON
        p_truco = float(1.0 / (1.0 + np.exp(-(mbon_act[3] - mbon_act[4]) * 2.0)))
        p_accept = float(1.0 / (1.0 + np.exp(-(mbon_act[5] - mbon_act[6]) * 2.0)))
        p_envido = float(1.0 / (1.0 + np.exp(-(mbon_act[7] - mbon_act[8]) * 2.0)))
        p_retruco = float(1.0 / (1.0 + np.exp(-(mbon_act[9] - mbon_act[10]) * 2.0)))
        p_vale4 = float(1.0 / (1.0 + np.exp(-(mbon_act[11] - mbon_act[12]) * 2.0)))
        p_acc_envido = float(1.0 / (1.0 + np.exp(-(mbon_act[13] - mbon_act[14]) * 2.0)))

        # Cartas
        exp_c = np.exp(mbon_act[:3] / 0.5)
        p_cards = (exp_c / np.sum(exp_c)).tolist()

        last_dop = float(self.dopamine_history[-1]) if self.dopamine_history else 0.0

        return {
            "num_kc": int(self.num_kc),
            "num_mbon": int(self.num_mbon),
            "active_kc_count": active_kc,
            "sparse_pct": round(pct_kc, 1),
            "kc_activity": kc_act.round(4).tolist(),
            "mbon_activity": mbon_act.round(4).tolist(),
            "inputs": state_vec.round(3).tolist(),
            "drives": {
                "truco": round(p_truco, 3),
                "retruco": round(p_retruco, 3),
                "vale4": round(p_vale4, 3),
                "accept_truco": round(p_accept, 3),
                "envido": round(p_envido, 3),
                "accept_envido": round(p_acc_envido, 3),
                "card_prefs": [round(p, 3) for p in p_cards]
            },
            "last_dopamine": round(last_dop, 2),
            "dopamine_history": [round(d, 2) for d in self.dopamine_history[-30:]]
        }

    def apply_dopamine_reward(self, dopamine_signal: float):
        """
        Regla de plasticidad sináptica de 3 factores (Drosophila Mushroom Body):
        Delta_W = lr * Dopamina * Elegibilidad
        - Dopamina > 0 (PAM): Recompensa por ganar manos, envido o truco.
        - Dopamina < 0 (PPL1): Castigo/aversión por perder o malas decisiones.
        """
        self.W += self.lr * dopamine_signal * self.eligibility
        # Mantener pesos sinápticos acotados biológicamente
        np.clip(self.W, 0.0, 2.0, out=self.W)
        # Reset gradual de elegibilidad
        self.eligibility *= 0.5
        self.dopamine_history.append(dopamine_signal)

    def save_model(self, filepath: str = "data/fly_truco_model.npz"):
        """Guarda la red neuronal entrenada de la mosca."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.savez(filepath, W=self.W, W_pn_kc=self.W_pn_kc)
        print(f"Modelo cerebral guardado en {filepath}")

    def load_model(self, filepath: str = "data/fly_truco_model.npz"):
        """Carga los pesos sinápticos aprendidos."""
        if Path(filepath).exists():
            data = np.load(filepath)
            self.W = data['W']
            self.W_pn_kc = data['W_pn_kc']
            print(f"Modelo cerebral cargado exitosamente desde {filepath}")
