import sys
from collections import defaultdict

# Use the project's parser and circuit utilities so depths match the optimizer's method
project_root = __import__('os').path.dirname(__import__('os').path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import parser as qparser
from src import circuits


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


def parse_qumode_qasm(filename):
    circ = qparser.read_qasm(filename)

    # latency-based depths per wire (uses Node.get_latecy())
    depths_map = circ.depths()

    # overall latency depth (max across all wires)
    overall_latency_depth = max(depths_map.values()) if depths_map else 0

    # unit-layer depth (every gate counts as 1 layer)
    unit_depth = compute_unit_depth(circ)

    # per-type (qubit/qumode) latency and unit depths
    qumode_latency_depths = [d for (w, d) in depths_map.items() if w[0] == 'qm']
    qubit_latency_depths = [d for (w, d) in depths_map.items() if w[0] == 'q']
    max_qm_latency = max(qumode_latency_depths) if qumode_latency_depths else 0
    max_qubit_latency = max(qubit_latency_depths) if qubit_latency_depths else 0

    # compute per-wire unit depths as in compute_unit_depth but per wire
    from collections import defaultdict
    wire_unit_depths = defaultdict(int)
    for node in circ.get_sequence():
        wires = getattr(node, 'wires', [])
        if not wires:
            continue
        if len(wires) >= 2:
            d = max(wire_unit_depths[w] for w in wires) + 1
            for w in wires:
                wire_unit_depths[w] = d
        else:
            wire_unit_depths[wires[0]] += 1

    qumode_unit_depths = [d for (w, d) in wire_unit_depths.items() if w[0] == 'qm']
    qubit_unit_depths = [d for (w, d) in wire_unit_depths.items() if w[0] == 'q']
    max_qm_unit = max(qumode_unit_depths) if qumode_unit_depths else 0
    max_qubit_unit = max(qubit_unit_depths) if qubit_unit_depths else 0

    # gate counts using existing circuit metrics
    metrics = circ.get_metrics()
    single_count = metrics.get('single op gate count', 0)
    multi_count = metrics.get('multi op gate count', 0)
    total_count = metrics.get('total gate count', single_count + multi_count)

    # counts of wires
    q_count = len([w for w in circ.wires if w[0] == 'q'])
    qm_count = len([w for w in circ.wires if w[0] == 'qm'])

    # print consolidated results
    print("Overall circuit depths:")
    print(f"  total latency depth: {overall_latency_depth}")
    print(f"  total unit-layer depth: {unit_depth}")
    print("")
    print("Per-subsystem depths (latency / unit layers):")
    print(f"  qumodes: {max_qm_latency} / {max_qm_unit}")
    print(f"  qubits: {max_qubit_latency} / {max_qubit_unit}")
    print("")
    print("Gate counts:")
    print(f"  single-op gates: {single_count}")
    print(f"  multi-op gates: {multi_count}")
    print(f"  total gates: {total_count}")
    print("")
    print(f"Number of qubits: {q_count}")
    print(f"Number of qumodes: {qm_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <cvdvqasm_file>")
        sys.exit(1)
    parse_qumode_qasm(sys.argv[1])
