#!/usr/bin/env python3
"""Simple peephole optimizer for CVDVQASM files.

This does a few local passes:
- cancel back-to-back self-inverse single-wire gates (h, x)
- cancel s <-> sdg pairs
- merge consecutive parameterized single-wire gates of the same type (e.g., R)

It uses the project's parser to read a circuit and the writer to emit optimized qasm.
"""
import sys
import os
# Ensure project root is on sys.path so `from src import ...` works when running the script
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import parser as qparser
from src import circuits


def peephole_optimize(circ: circuits.Circuit) -> circuits.Circuit:
    seq = circ.get_sequence()
    new_nodes = []
    # Track last node object for each wire so we can look back per-wire even when
    # operations on other wires are interleaved. Using node objects avoids
    # index-shift issues when removing earlier nodes from new_nodes.
    last_node_by_wire: dict = {}

    def find_prev_node_for_wire(wire, before_node=None):
        # scan backwards to find previous node object that touches 'wire'
        # If before_node is provided, search nodes earlier than its position.
        limit = len(new_nodes)
        if before_node is not None and before_node in new_nodes:
            limit = new_nodes.index(before_node)
        for node in reversed(new_nodes[:limit]):
            if wire in getattr(node, 'wires', []):
                return node
        return None

    for n in seq:
        # operate only on Gate-like nodes
        if not hasattr(n, 'nodeType'):
            new_nodes.append(n.copy_disconnected())
            # update last node for wires touched by this node
            for w in getattr(n, 'wires', []):
                last_node_by_wire[w] = new_nodes[-1]
            continue

        if len(n.wires) == 1:
            wire = n.wires[0]
            prev_node = last_node_by_wire.get(wire, None)
            if prev_node is not None and prev_node in new_nodes:
                last = prev_node
                # ensure last is gate-like and single-wire
                if hasattr(last, 'nodeType') and len(getattr(last, 'wires', [])) == 1:
                    # cancel h/h and x/x
                    if last.nodeType == n.nodeType and last.nodeType in ('h', 'x'):
                        # remove the previous node object
                        new_nodes.remove(last)
                        # update last_node_by_wire for this wire to its previous occurrence
                        prev_prev = find_prev_node_for_wire(wire)
                        if prev_prev is None:
                            last_node_by_wire.pop(wire, None)
                        else:
                            last_node_by_wire[wire] = prev_prev
                        continue
                    # cancel s and sdg
                    if (last.nodeType, n.nodeType) in (('s', 'sdg'), ('sdg', 's')):
                        new_nodes.remove(last)
                        prev_prev = find_prev_node_for_wire(wire)
                        if prev_prev is None:
                            last_node_by_wire.pop(wire, None)
                        else:
                            last_node_by_wire[wire] = prev_prev
                        continue
                    # merge parameterized gates, e.g., R + R -> R(sum)
                    if last.nodeType == n.nodeType and getattr(last, 'params', None) and getattr(n, 'params', None):
                        try:
                            new_val = complex(last.params[0]) + complex(n.params[0])
                            last.params[0] = new_val
                            # last remains the last for this wire
                            continue
                        except Exception:
                            pass

            # otherwise append a copy
            new_node = n.copy_disconnected()
            new_nodes.append(new_node)
            last_node_by_wire[wire] = new_node
        else:
            # multi-wire gates: append and update last index for all wires
            new_node = n.copy_disconnected()
            new_nodes.append(new_node)
            for w in n.wires:
                last_node_by_wire[w] = new_node

    # build new circuit with same wires
    new_circ = circuits.Circuit(wires=circ.wires.copy())
    for node in new_nodes:
        new_circ.append_node(node)
    return new_circ


def main(fname_in: str, fname_out: str | None = None):
    circ = qparser.read_qasm(fname_in)
    before = circ.get_metrics()
    before_depths = circ.depths()
    print("Before metrics:")
    for k, v in before.items():
        print(f"  {k}: {v}")
    # compute unit-layer depth (every gate costs 1 layer)
    def compute_unit_depth(circ_obj: circuits.Circuit) -> int:
        from collections import defaultdict
        depths = defaultdict(int)
        for node in circ_obj.get_sequence():
            wires = getattr(node, 'wires', [])
            if not wires:
                continue
            if len(wires) >= 2:
                d = max(depths[w] for w in wires) + 1
                for w in wires:
                    depths[w] = d
            else:
                depths[wires[0]] += 1
        return max(depths.values()) if depths else 0

    before_unit_depth = compute_unit_depth(circ)
    print(f"  depth_layers (unit cost): {before_unit_depth}")

    optimized = peephole_optimize(circ)
    after = optimized.get_metrics()
    after_depths = optimized.depths()
    after_unit_depth = compute_unit_depth(optimized)
    print("After metrics:")
    for k, v in after.items():
        print(f"  {k}: {v}")
    print(f"  depth_layers (unit cost): {after_unit_depth}")

    if fname_out is None:
        if fname_in.endswith('.cvdvqasm'):
            fname_out = fname_in.replace('.cvdvqasm', '_opt.cvdvqasm')
        else:
            fname_out = fname_in + '.opt'

    qparser.write_qasm(optimized, fname_out)
    print(f"Wrote optimized circuit to: {fname_out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.cvdvqasm> [<output.cvdvqasm>]")
        sys.exit(1)
    fin = sys.argv[1]
    fout = sys.argv[2] if len(sys.argv) > 2 else None
    main(fin, fout)
