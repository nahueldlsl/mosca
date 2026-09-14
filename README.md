# 🧠 FlyEM Male CNS v1.0 - Exploración del Cerebro de la Mosca y FlyTruco 🇺🇾

Este proyecto contiene herramientas y datos para analizar el conectoma completo y la microscopía electrónica (EM) del cerebro de la mosca (*Drosophila melanogaster*, Male CNS v1.0 de HHMI Janelia / MRC LMB / Cambridge / Google Research), además de un agente bio-inspirado que aprende a jugar al **Truco Uruguayo** usando el circuito del Mushroom Body.

---

## 📁 Estructura del Proyecto

```
mosca/
├── .gitignore
├── requirements.txt
├── README.md
├── demo_fly_brain.ipynb           # Notebook interactivo con análisis y demo de FlyTruco
├── truco_engine.py                # Motor completo de reglas de Truco Uruguayo (muestra, piezas, matas)
├── fly_brain_agent.py             # Agente neuronal basado en el Mushroom Body real (KC -> MBON)
├── train_fly_truco.py             # Entrenamiento por refuerzo dopaminérgico (PAM/PPL1)
├── play_truco_interactive.py      # Duelo interactivo por consola: Humano vs Cerebro de la Mosca
├── explore_connectome.py          # Consultas de conectoma, neurotransmisores y circuitos
├── fetch_em_cutout.py             # Extractor de cortes y cubos 3D de microscopía EM
├── fetch_neuron_skeleton.py       # Descarga y renderizado 3D de morfologías SWC
├── download_connectome.py         # Descargador del conectoma
└── data/
    ├── flat-connectome/           # 11 tablas .feather del conectoma (29.17 GB)
    ├── skeletons/                 # Esqueletos 3D (.swc y .png)
    ├── em_cutouts/                # Muestras de microscopía (.png y .npy)
    ├── fly_truco_model.npz        # Pesos sinápticos entrenados de la mosca
    └── fly_truco_learning_curve.png # Gráfica de aprendizaje y dopamina
```

---

## 🇺🇾 FlyTruco: Duelo Humano vs Mosca

### 1. Panel Visual Interactivo Web (Recomendado):
```bash
python run_dashboard.py
```
*(O haz doble clic en `run_dashboard.bat` en Windows)*
- Abre una aplicación web moderna en tu navegador (`http://localhost:8000`).
- **Mesa de Juego**: Con cartas españolas visuales, Muestra, Piezas destacadas y botones para cantar Truco, **Re-truco**, **Vale 4**, Envido o irse al mazo.
- **Visualizador Neuronal en Tiempo Real**: Canvas con las **500 Células de Kenyon (KC)** iluminándose en vivo según el disparo disperso (~10%), medidores de impulso MBON y modulación dopaminérgica (PAM/PPL1).
- **Gimnasio de Entrenamiento**: Permite entrenar al cerebro de la mosca en vivo con un solo clic y ver sus curvas de aprendizaje y dopamina.

### 2. Duelo Interactivo por Consola:
```bash
python play_truco_interactive.py
```
- Duelo en terminal con telemetría neuronal en vivo, reglamento completo con **Truco**, **Re-truco** y **Vale 4**.

### 3. Entrenar el cerebro de la mosca por consola:
```bash
python train_fly_truco.py --episodes 600
```
- Entrena al agente jugando manos con la escalera completa de apuestas contra el bot heurístico.
- Guarda los pesos en `data/fly_truco_model.npz` y las curvas en `data/fly_truco_learning_curve.png`.

---

## 🔬 Exploración del Conectoma y Microscopía

### Resumen biológico del conectoma:
```bash
python explore_connectome.py --summary
```

### Consultar una neurona y su circuito:
```bash
python explore_connectome.py --query 10112 --subgraph 10112
```

### Extraer corte de microscopía electrónica a 8nm:
```bash
python fetch_em_cutout.py --x 40000 --y 40000 --z 20000 --size 500
```

### Descargar y visualizar esqueleto 3D:
```bash
python fetch_neuron_skeleton.py --body 10112
```

---

## 📓 Notebook Interactivo
Abre `demo_fly_brain.ipynb` en tu IDE (VS Code, Cursor, Jupyter) para probar todas las funciones visualmente.
