"""
Script para explorar, consultar y analizar el conectoma del cerebro de la mosca (FlyEM Male CNS v1.0).
Permite buscar neuronas por tipo, consultar neurotransmisores, analizar conectividad y circuitos.
"""

import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("data", "flat-connectome")
ANN_FILE = os.path.join(DATA_DIR, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
NT_FILE = os.path.join(DATA_DIR, "body-neurotransmitters-male-cns-v1.0.feather")
WEIGHTS_FILE = os.path.join(DATA_DIR, "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather")

def load_data():
    """Carga los DataFrames principales del conectoma."""
    print("Cargando tablas del conectoma...")
    if not os.path.exists(ANN_FILE):
        raise FileNotFoundError(f"No se encontró {ANN_FILE}. Ejecuta primero `python download_connectome.py`.")

    ann = pd.read_feather(ANN_FILE)
    nt = pd.read_feather(NT_FILE) if os.path.exists(NT_FILE) else None
    weights = pd.read_feather(WEIGHTS_FILE) if os.path.exists(WEIGHTS_FILE) else None
    
    print(f" - Neuronas anotadas: {len(ann):,}")
    if nt is not None:
        print(f" - Predicciones de neurotransmisores: {len(nt):,}")
    if weights is not None:
        print(f" - Conexiones sinápticas (aristas): {len(weights):,}")
    return ann, nt, weights

def show_summary(ann, nt, weights):
    """Muestra estadísticas generales del conectoma."""
    print("\n" + "=" * 60)
    print(" RESUMEN GENERAL DEL CEREBRO DE LA MOSCA (Male CNS v1.0)")
    print("=" * 60)
    print(f"Total de cuerpos anotados: {len(ann):,}")
    
    if "superclass" in ann.columns:
        print("\nDistribución por Superclase neuronal:")
        sc_counts = ann["superclass"].value_counts().dropna().head(10)
        for sc, cnt in sc_counts.items():
            print(f"  {sc:<25}: {cnt:>7,}")

    if "cell_type" in ann.columns or "type" in ann.columns:
        col = "type" if "type" in ann.columns else "cell_type"
        n_types = ann[col].nunique()
        print(f"\nNúmero de tipos celulares únicos catalogados: {n_types:,}")

    if nt is not None and "predicted_nt" in nt.columns:
        print("\nDistribución de Neurotransmisores predichos:")
        nt_counts = nt["predicted_nt"].value_counts().head(10)
        for name, cnt in nt_counts.items():
            pct = (cnt / len(nt)) * 100
            print(f"  {name:<20}: {cnt:>8,} ({pct:.1f}%)")

    if weights is not None:
        print("\nEstadísticas de la Red Sináptica:")
        print(f"  Total conexiones: {len(weights):,}")
        print(f"  Peso sináptico total: {weights['weight'].sum():,}")
        print(f"  Peso sináptico promedio: {weights['weight'].mean():.2f}")
        print(f"  Peso sináptico máximo en una conexión: {weights['weight'].max()}")

def query_neuron(body_id: int, ann, nt, weights, top_k: int = 5):
    """Consulta los detalles, neurotransmisor y conexiones de una neurona."""
    print("\n" + "=" * 60)
    print(f" CONSULTA DE NEURONA: Body ID = {body_id}")
    print("=" * 60)

    # 1. Anotaciones
    neuron_ann = ann[ann["bodyId"] == body_id]
    if neuron_ann.empty:
        print(f"No se encontró la neurona {body_id} en las anotaciones.")
    else:
        row = neuron_ann.iloc[0]
        print(f"Tipo:       {row.get('type', 'N/A')}")
        print(f"Instancia:  {row.get('instance', 'N/A')}")
        print(f"Superclase: {row.get('superclass', 'N/A')}")
        print(f"Lado:       {row.get('somaSide', 'N/A')}")
        print(f"Grupo:      {row.get('group', 'N/A')}")

    # 2. Neurotransmisor
    if nt is not None:
        neuron_nt = nt[nt["body"] == body_id]
        if not neuron_nt.empty:
            nt_row = neuron_nt.iloc[0]
            print(f"Neurotransmisor predicho: {nt_row.get('predicted_nt', 'N/A')} (confianza: {nt_row.get('predicted_nt_confidence', 'N/A')})")

    # 3. Conexiones
    if weights is not None:
        # Entradas (upstream): quién le envía sinapsis a esta neurona
        upstream = weights[weights["body_post"] == body_id].sort_values("weight", ascending=False).head(top_k)
        print(f"\nTop {top_k} entradas sinápticas (Upstream -> {body_id}):")
        if upstream.empty:
            print("  (Ninguna)")
        else:
            for _, r in upstream.iterrows():
                print(f"  Desde {r['body_pre']} ({r.get('type_pre', 'N/A')}): {r['weight']} sinapsis")

        # Salidas (downstream): a quién envía sinapsis esta neurona
        downstream = weights[weights["body_pre"] == body_id].sort_values("weight", ascending=False).head(top_k)
        print(f"\nTop {top_k} salidas sinápticas ({body_id} -> Downstream):")
        if downstream.empty:
            print("  (Ninguna)")
        else:
            for _, r in downstream.iterrows():
                print(f"  Hacia {r['body_post']} ({r.get('type_post', 'N/A')}): {r['weight']} sinapsis")

def build_local_subgraph(body_id: int, weights, ann, depth: int = 1, min_weight: int = 5, out_dir: str = "data"):
    """Construye y visualiza un subgrafo local de conectividad alrededor de una neurona."""
    os.makedirs(out_dir, exist_ok=True)
    G = nx.DiGraph()

    # Filtrar conexiones relevantes
    direct = weights[((weights["body_pre"] == body_id) | (weights["body_post"] == body_id)) & (weights["weight"] >= min_weight)]
    if direct.empty:
        print(f"No hay conexiones con peso >= {min_weight} para la neurona {body_id}")
        return

    # Añadir aristas
    for _, row in direct.iterrows():
        G.add_edge(row["body_pre"], row["body_post"], weight=row["weight"])

    # Añadir atributos
    ann_dict = ann.set_index("bodyId")["type"].to_dict() if "type" in ann.columns else {}
    labels = {node: f"{ann_dict.get(node, node)}\n({node})" if node == body_id else ann_dict.get(node, str(node)) for node in G.nodes()}

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ["#ff4444" if n == body_id else "#88ccff" for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=15, width=1.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(f"Circuito Local para Neurona {body_id} (peso >= {min_weight})")
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"subgraph_{body_id}.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"\nGrafo del circuito local guardado en: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Explorador del conectoma de FlyEM Male CNS v1.0")
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen general del conectoma")
    parser.add_argument("--query", type=int, help="Consultar información de una neurona por bodyId")
    parser.add_argument("--subgraph", type=int, help="Generar imagen del subgrafo local de una neurona")
    parser.add_argument("--min-weight", type=int, default=10, help="Peso mínimo para el subgrafo (default: 10)")
    args = parser.parse_args()

    ann, nt, weights = load_data()

    if args.query:
        query_neuron(args.query, ann, nt, weights)
        if args.subgraph:
            build_local_subgraph(args.query, weights, ann, min_weight=args.min_weight)
    elif args.subgraph:
        build_local_subgraph(args.subgraph, weights, ann, min_weight=args.min_weight)
    else:
        show_summary(ann, nt, weights)
        # Mostrar un ejemplo con una neurona interesante
        sample_body = ann.iloc[100]["bodyId"]
        query_neuron(sample_body, ann, nt, weights)

if __name__ == "__main__":
    main()
