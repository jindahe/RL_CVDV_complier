"""
Gate library for hybrid CV-DV quantum compilation.

This module defines the gate set available for compiling target Hamiltonians.
Gates are parameterized and can be used as actions in the RL environment.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto

from .operators import (
    Operator, tensor, 
    pauli_x, pauli_y, pauli_z, qubit_identity,
    rotation_x, rotation_y, rotation_z, hadamard,
    sigma_plus, sigma_minus,
    creation, annihilation, number_op, position, momentum, qumode_identity,
    displacement, squeeze, rotation_mode,
    jaynes_cummings, anti_jaynes_cummings, dispersive_coupling,
    conditional_displacement
)


class GateType(Enum):
    """Enumeration of gate types in the hybrid system."""
    # Single-qubit gates
    RX = auto()
    RY = auto()
    RZ = auto()
    HADAMARD = auto()
    
    # Single-qumode gates
    DISPLACEMENT = auto()
    SQUEEZE = auto()
    PHASE_ROTATION = auto()
    
    # Hybrid (qubit-qumode) gates
    JAYNES_CUMMINGS = auto()
    ANTI_JAYNES_CUMMINGS = auto()
    DISPERSIVE = auto()
    CONDITIONAL_DISPLACEMENT = auto()
    
    # Identity (no-op)
    IDENTITY = auto()


@dataclass
class GateDefinition:
    """
    Definition of a quantum gate.
    
    Attributes:
        gate_type: Type of the gate
        name: Human-readable name
        n_params: Number of continuous parameters
        param_ranges: List of (min, max) tuples for each parameter
        is_hybrid: Whether this gate couples qubit and qumode
    """
    gate_type: GateType
    name: str
    n_params: int
    param_ranges: List[Tuple[float, float]]
    is_hybrid: bool
    
    def __hash__(self):
        return hash(self.gate_type)


# Define all available gates
GATE_DEFINITIONS: Dict[GateType, GateDefinition] = {
    # Single-qubit gates
    GateType.RX: GateDefinition(
        GateType.RX, "Rx", 1, [(-np.pi, np.pi)], False
    ),
    GateType.RY: GateDefinition(
        GateType.RY, "Ry", 1, [(-np.pi, np.pi)], False
    ),
    GateType.RZ: GateDefinition(
        GateType.RZ, "Rz", 1, [(-np.pi, np.pi)], False
    ),
    GateType.HADAMARD: GateDefinition(
        GateType.HADAMARD, "H", 0, [], False
    ),
    
    # Single-qumode gates
    GateType.DISPLACEMENT: GateDefinition(
        GateType.DISPLACEMENT, "D", 2, [(-2.0, 2.0), (-np.pi, np.pi)], False
    ),  # amplitude r and phase phi -> alpha = r*exp(i*phi)
    GateType.SQUEEZE: GateDefinition(
        GateType.SQUEEZE, "S", 2, [(0.0, 1.5), (-np.pi, np.pi)], False
    ),  # squeezing r and phase phi
    GateType.PHASE_ROTATION: GateDefinition(
        GateType.PHASE_ROTATION, "R", 1, [(-np.pi, np.pi)], False
    ),
    
    # Hybrid gates
    GateType.JAYNES_CUMMINGS: GateDefinition(
        GateType.JAYNES_CUMMINGS, "JC", 1, [(0.0, np.pi)], True
    ),  # evolution time gt
    GateType.ANTI_JAYNES_CUMMINGS: GateDefinition(
        GateType.ANTI_JAYNES_CUMMINGS, "AJC", 1, [(0.0, np.pi)], True
    ),
    GateType.DISPERSIVE: GateDefinition(
        GateType.DISPERSIVE, "Disp", 1, [(-np.pi, np.pi)], True
    ),  # chi*t
    GateType.CONDITIONAL_DISPLACEMENT: GateDefinition(
        GateType.CONDITIONAL_DISPLACEMENT, "CD", 2, [(-2.0, 2.0), (-np.pi, np.pi)], True
    ),
    
    # Identity
    GateType.IDENTITY: GateDefinition(
        GateType.IDENTITY, "I", 0, [], False
    ),
}


class Gate:
    """
    A quantum gate instance with specific parameters.
    
    Attributes:
        gate_type: Type of gate
        params: Parameter values
        qumode_dim: Dimension of the qumode Fock space
    """
    
    def __init__(self, gate_type: GateType, params: List[float], qumode_dim: int):
        """
        Initialize a gate.
        
        Args:
            gate_type: Type of gate from GateType enum
            params: List of parameter values
            qumode_dim: Fock space truncation dimension
        """
        self.gate_type = gate_type
        self.params = params
        self.qumode_dim = qumode_dim
        self.definition = GATE_DEFINITIONS[gate_type]
        
        if len(params) != self.definition.n_params:
            raise ValueError(
                f"Gate {gate_type.name} expects {self.definition.n_params} params, "
                f"got {len(params)}"
            )
    
    def to_operator(self) -> Operator:
        """
        Convert the gate to a unitary operator in the hybrid space.
        
        Returns:
            Operator in the (qubit ⊗ qumode) Hilbert space
        """
        dim = self.qumode_dim
        
        if self.gate_type == GateType.RX:
            qubit_op = rotation_x(self.params[0])
            return tensor(qubit_op, qumode_identity(dim))
        
        elif self.gate_type == GateType.RY:
            qubit_op = rotation_y(self.params[0])
            return tensor(qubit_op, qumode_identity(dim))
        
        elif self.gate_type == GateType.RZ:
            qubit_op = rotation_z(self.params[0])
            return tensor(qubit_op, qumode_identity(dim))
        
        elif self.gate_type == GateType.HADAMARD:
            return tensor(hadamard(), qumode_identity(dim))
        
        elif self.gate_type == GateType.DISPLACEMENT:
            r, phi = self.params
            alpha = r * np.exp(1j * phi)
            qumode_op = displacement(dim, alpha)
            return tensor(qubit_identity(), qumode_op)
        
        elif self.gate_type == GateType.SQUEEZE:
            r, phi = self.params
            qumode_op = squeeze(dim, r, phi)
            return tensor(qubit_identity(), qumode_op)
        
        elif self.gate_type == GateType.PHASE_ROTATION:
            qumode_op = rotation_mode(dim, self.params[0])
            return tensor(qubit_identity(), qumode_op)
        
        elif self.gate_type == GateType.JAYNES_CUMMINGS:
            # U = exp(-i * gt * H_JC)
            H_JC = jaynes_cummings(dim, g=1.0)
            return H_JC.exp(-1j * self.params[0])
        
        elif self.gate_type == GateType.ANTI_JAYNES_CUMMINGS:
            H_AJC = anti_jaynes_cummings(dim, g=1.0)
            return H_AJC.exp(-1j * self.params[0])
        
        elif self.gate_type == GateType.DISPERSIVE:
            H_disp = dispersive_coupling(dim, chi=1.0)
            return H_disp.exp(-1j * self.params[0])
        
        elif self.gate_type == GateType.CONDITIONAL_DISPLACEMENT:
            r, phi = self.params
            alpha = r * np.exp(1j * phi)
            return conditional_displacement(dim, alpha)
        
        elif self.gate_type == GateType.IDENTITY:
            return tensor(qubit_identity(), qumode_identity(dim))
        
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")
    
    def __repr__(self) -> str:
        param_str = ", ".join(f"{p:.3f}" for p in self.params)
        return f"{self.definition.name}({param_str})"


class GateLibrary:
    """
    Library of available gates for the compiler.
    
    This defines the action space for the RL agent.
    """
    
    def __init__(
        self,
        qumode_dim: int = 10,
        gate_types: Optional[List[GateType]] = None,
        n_discrete_params: int = 5
    ):
        """
        Initialize the gate library.
        
        Args:
            qumode_dim: Fock space truncation dimension
            gate_types: List of gate types to include (None = all)
            n_discrete_params: Number of discrete values for each parameter
        """
        self.qumode_dim = qumode_dim
        self.n_discrete_params = n_discrete_params
        
        if gate_types is None:
            self.gate_types = list(GATE_DEFINITIONS.keys())
        else:
            self.gate_types = gate_types
        
        # Build the discrete action space
        self._build_action_space()
    
    def _build_action_space(self):
        """Build the discrete action space by discretizing parameters."""
        self.actions: List[Tuple[GateType, List[float]]] = []
        
        for gate_type in self.gate_types:
            definition = GATE_DEFINITIONS[gate_type]
            
            if definition.n_params == 0:
                # No parameters
                self.actions.append((gate_type, []))
            else:
                # Discretize each parameter
                param_values = []
                for min_val, max_val in definition.param_ranges:
                    values = np.linspace(min_val, max_val, self.n_discrete_params)
                    param_values.append(values)
                
                # Create all combinations
                from itertools import product
                for params in product(*param_values):
                    self.actions.append((gate_type, list(params)))
        
        self.n_actions = len(self.actions)
    
    def get_gate(self, action_idx: int) -> Gate:
        """
        Get a Gate instance for a given action index.
        
        Args:
            action_idx: Index into the action space
        
        Returns:
            Gate instance
        """
        gate_type, params = self.actions[action_idx]
        return Gate(gate_type, params, self.qumode_dim)
    
    def get_action_idx(self, gate_type: GateType, params: List[float]) -> int:
        """
        Find the action index closest to the given gate parameters.
        
        Args:
            gate_type: Type of gate
            params: Parameter values
        
        Returns:
            Closest action index
        """
        best_idx = 0
        best_dist = float('inf')
        
        for idx, (gt, p) in enumerate(self.actions):
            if gt != gate_type:
                continue
            
            dist = sum((a - b) ** 2 for a, b in zip(params, p))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        return best_idx
    
    def sample_random_action(self) -> int:
        """Sample a random action index."""
        return np.random.randint(self.n_actions)
    
    def sample_random_gate(self) -> Gate:
        """Sample a random gate."""
        return self.get_gate(self.sample_random_action())
    
    def __len__(self) -> int:
        return self.n_actions
    
    def __repr__(self) -> str:
        return f"GateLibrary(n_actions={self.n_actions}, gate_types={len(self.gate_types)})"


class GateSequence:
    """
    A sequence of quantum gates.
    
    This represents a compiled quantum circuit.
    """
    
    def __init__(self, gates: Optional[List[Gate]] = None):
        """
        Initialize a gate sequence.
        
        Args:
            gates: Initial list of gates (empty if None)
        """
        self.gates = gates if gates is not None else []
    
    def append(self, gate: Gate) -> "GateSequence":
        """Add a gate to the sequence."""
        self.gates.append(gate)
        return self
    
    def to_operator(self) -> Operator:
        """
        Convert the gate sequence to a single unitary operator.
        
        Gates are applied left to right: U = U_n @ ... @ U_2 @ U_1
        
        Returns:
            Combined unitary operator
        """
        if len(self.gates) == 0:
            raise ValueError("Cannot convert empty sequence to operator")
        
        result = self.gates[0].to_operator()
        for gate in self.gates[1:]:
            result = gate.to_operator() @ result
        
        return result
    
    def depth(self) -> int:
        """Return the circuit depth (number of gates)."""
        return len(self.gates)
    
    def copy(self) -> "GateSequence":
        """Create a copy of this sequence."""
        return GateSequence(self.gates.copy())
    
    def __len__(self) -> int:
        return len(self.gates)
    
    def __repr__(self) -> str:
        if len(self.gates) <= 5:
            gate_str = " -> ".join(str(g) for g in self.gates)
        else:
            gate_str = " -> ".join(str(g) for g in self.gates[:3])
            gate_str += f" -> ... -> {self.gates[-1]}"
        return f"GateSequence[{gate_str}]"
    
    def to_string(self) -> str:
        """Convert to a detailed string representation."""
        lines = ["Gate Sequence:"]
        for i, gate in enumerate(self.gates):
            lines.append(f"  {i+1}. {gate}")
        return "\n".join(lines)
