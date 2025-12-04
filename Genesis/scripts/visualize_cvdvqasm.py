#!/usr/bin/env python3
"""
Render a CVDVQASM file (logical or physical) into a simple circuit diagram (PNG).

Usage:
  python scripts/visualize_cvdvqasm.py --input output/electronicVibration_small_Result.cvdvqasm [--out image.png] [--hide-qm]

The renderer draws horizontal wires for q (qubit) and qm (qumode) and places
gate boxes in sequence order left-to-right. Multi-wire gates are connected with
a vertical line and labeled at the top wire.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple

# Ensure project root is importable when running as a script
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import src.parser as cvdparser
import src.circuits as circuits

Wire = Tuple[str, int]  # ('q'|'qm', index)


def load_circuit(qasm_path: str) -> circuits.Circuit:
    if not os.path.exists(qasm_path):
        raise FileNotFoundError(f"File not found: {qasm_path}")
    # Preprocess: remove qreg header (parser can grow wires dynamically),
    # and rewrite qreg if needed. This avoids parser.py's strict int assumption.
    import tempfile
    import re
    with open(qasm_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    qreg_pat = re.compile(r'^\s*qreg\b', re.IGNORECASE)
    for line in lines:
        if qreg_pat.match(line):
            # drop qreg line; parser will extend wires on-the-fly
            continue
        new_lines.append(line)
    # Write to temp file and parse
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.cvdvqasm') as tf:
        tf.writelines(new_lines)
        tf.flush()
        circ = cvdparser.read_qasm(tf.name)
    # Clean up temp file if desired (optional)
    return circ


def sort_wires(wires: List[Wire], include_qm: bool = True) -> List[Wire]:
    q = sorted([w for w in wires if w[0] == 'q'], key=lambda x: x[1])
    qm = sorted([w for w in wires if w[0] == 'qm'], key=lambda x: x[1]) if include_qm else []
    return q + qm


def gate_color(node_type: str) -> str:
    palette = {
        'h': '#4e79a7',
        's': '#a0cbe8',
        'sdg': '#59a14f',
        'R': '#f28e2b',
        'D': '#e15759',
        'BS': '#b07aa1',
        'CP': '#edc948',
        'CD': '#76b7b2',
        'pauliNode': '#ff9da7',
        'commentLine': '#cccccc',
    }
    return palette.get(node_type, '#8cd17d')


def format_wire(w: Wire) -> str:
    t, i = w
    return f"{t}[{i}]"


def format_label(node: circuits.Node) -> str:
    t = node.nodeType
    if hasattr(node, 'params') and node.params:
        # Show compact params, strip parentheses in string repr
        params = ",".join(str(p).strip("()") for p in node.params)
        return f"{t}({params})"
    if t == 'pauliNode':
        try:
            s = getattr(node, 'pauli_string')
            disp = getattr(node, 'displacement')
            return f"pauli({str(disp).strip()})\n{s}"
        except Exception:
            return "pauli"
    return t


def render_circuit(circ: circuits.Circuit, out_path: str, title: str | None = None, include_qm: bool = True) -> None:
    wires = sort_wires(circ.wires, include_qm=include_qm)
    seq = circ.get_sequence()

    # Layout params
    n_wires = len(wires)
    n_cols = max(1, len(seq))
    cell_w = 0.9
    cell_h = 0.6
    margin_x = 1.0
    margin_y = 0.8
    fig_w = max(6.0, margin_x * 2 + n_cols * cell_w)
    fig_h = max(3.0, margin_y * 2 + n_wires * cell_h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # Map wires to y positions (top to bottom)
    y_map: Dict[Wire, float] = {}
    for idx, w in enumerate(wires):
        y = fig_h - margin_y - (idx + 0.5) * cell_h
        y_map[w] = y
        # Draw wire baseline
        ax.hlines(y, margin_x, fig_w - margin_x, colors="#aaaaaa", linewidth=1)
        ax.text(margin_x - 0.4, y, format_wire(w), va='center', ha='right', fontsize=9, family='monospace')

    # Draw gates left to right
    for t_idx, node in enumerate(seq):
        involved = [w for w in node.wires if w in y_map]
        if not involved:
            continue
        x = margin_x + (t_idx + 0.5) * cell_w
        ys = [y_map[w] for w in involved]
        y_top = max(ys)
        y_bot = min(ys)
        col = gate_color(node.nodeType)

        # Multi-wire: draw vertical connector
        if len(involved) > 1:
            ax.vlines(x, y_bot, y_top, colors=col, linewidth=2, alpha=0.9)

        # For each wire, draw a small box marker at (x, y)
        for w in involved:
            y = y_map[w]
            rect = plt.Rectangle((x - cell_w * 0.35, y - cell_h * 0.2), cell_w * 0.7, cell_h * 0.4,
                                 facecolor=col, edgecolor='black', linewidth=0.7, alpha=0.95)
            ax.add_patch(rect)

        # Label once near the top wire to avoid clutter
        label = format_label(node)
        ax.text(x, y_top + cell_h * 0.25, label, ha='center', va='bottom', fontsize=8, family='monospace')

    if title:
        ax.set_title(title, fontsize=12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize a CVDVQASM file to a circuit PNG.")
    ap.add_argument('--input', '-i', required=True, help='Path to .cvdvqasm file (logical or physical).')
    ap.add_argument('--out', '-o', help='Output image path (.png). Defaults to alongside input.')
    ap.add_argument('--title', help='Optional title on the figure.')
    ap.add_argument('--hide-qm', action='store_true', help='Hide qumode (qm) wires in the diagram.')
    args = ap.parse_args()

    circ = load_circuit(args.input)
    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(args.input)
        out_path = base + '.png'

    title = args.title or os.path.basename(args.input)
    render_circuit(circ, out_path, title=title, include_qm=not args.hide_qm)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
