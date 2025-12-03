"""
Tests for gates module.
"""

import numpy as np
import pytest
from hybrid_compiler.gates import (
    Gate, GateType, GateDefinition, GateLibrary, GateSequence,
    GATE_DEFINITIONS
)


class TestGateDefinitions:
    """Test suite for gate definitions."""
    
    def test_all_gate_types_defined(self):
        """Test that all GateType values have definitions."""
        for gate_type in GateType:
            assert gate_type in GATE_DEFINITIONS
    
    def test_definition_consistency(self):
        """Test that definitions are self-consistent."""
        for gate_type, definition in GATE_DEFINITIONS.items():
            assert definition.gate_type == gate_type
            assert definition.n_params == len(definition.param_ranges)


class TestGate:
    """Test suite for Gate class."""
    
    def test_gate_creation(self):
        """Test gate instantiation."""
        gate = Gate(GateType.RX, [np.pi/4], qumode_dim=5)
        
        assert gate.gate_type == GateType.RX
        assert len(gate.params) == 1
        assert gate.qumode_dim == 5
    
    def test_gate_wrong_params(self):
        """Test that wrong number of parameters raises error."""
        with pytest.raises(ValueError):
            Gate(GateType.RX, [np.pi/4, np.pi/2], qumode_dim=5)  # RX needs 1 param
    
    def test_gate_to_operator_unitary(self):
        """Test that gates produce unitary operators."""
        qumode_dim = 5
        
        test_gates = [
            Gate(GateType.RX, [np.pi/4], qumode_dim),
            Gate(GateType.RY, [np.pi/3], qumode_dim),
            Gate(GateType.RZ, [np.pi/2], qumode_dim),
            Gate(GateType.HADAMARD, [], qumode_dim),
            Gate(GateType.DISPLACEMENT, [0.3, np.pi/4], qumode_dim),
            Gate(GateType.SQUEEZE, [0.2, 0.0], qumode_dim),
            Gate(GateType.PHASE_ROTATION, [np.pi/6], qumode_dim),
            Gate(GateType.IDENTITY, [], qumode_dim),
        ]
        
        for gate in test_gates:
            U = gate.to_operator()
            # U†U = I
            product = U.dag() @ U
            np.testing.assert_array_almost_equal(
                product.matrix,
                np.eye(2 * qumode_dim, dtype=complex),
                decimal=5,
                err_msg=f"Gate {gate} should be unitary"
            )
    
    def test_hybrid_gates_unitary(self):
        """Test that hybrid gates are unitary."""
        qumode_dim = 8
        
        test_gates = [
            Gate(GateType.JAYNES_CUMMINGS, [np.pi/4], qumode_dim),
            Gate(GateType.ANTI_JAYNES_CUMMINGS, [np.pi/4], qumode_dim),
            Gate(GateType.DISPERSIVE, [np.pi/4], qumode_dim),
            Gate(GateType.CONDITIONAL_DISPLACEMENT, [0.3, 0.0], qumode_dim),
        ]
        
        for gate in test_gates:
            U = gate.to_operator()
            product = U.dag() @ U
            np.testing.assert_array_almost_equal(
                product.matrix,
                np.eye(2 * qumode_dim, dtype=complex),
                decimal=5,
                err_msg=f"Hybrid gate {gate} should be unitary"
            )
    
    def test_identity_gate(self):
        """Test that identity gate is actually identity."""
        qumode_dim = 5
        gate = Gate(GateType.IDENTITY, [], qumode_dim)
        
        U = gate.to_operator()
        np.testing.assert_array_almost_equal(
            U.matrix, np.eye(2 * qumode_dim, dtype=complex)
        )
    
    def test_gate_repr(self):
        """Test gate string representation."""
        gate = Gate(GateType.RX, [np.pi/4], qumode_dim=5)
        
        repr_str = repr(gate)
        assert "Rx" in repr_str
        assert "0.785" in repr_str  # π/4


class TestGateLibrary:
    """Test suite for GateLibrary."""
    
    def test_library_creation(self):
        """Test library instantiation."""
        library = GateLibrary(qumode_dim=5, n_discrete_params=3)
        
        assert library.qumode_dim == 5
        assert library.n_discrete_params == 3
        assert library.n_actions > 0
    
    def test_library_subset_gates(self):
        """Test library with subset of gate types."""
        gate_types = [GateType.RX, GateType.RY, GateType.IDENTITY]
        library = GateLibrary(qumode_dim=5, gate_types=gate_types, n_discrete_params=3)
        
        # Should only have actions for specified gates
        for gate_type, params in library.actions:
            assert gate_type in gate_types
    
    def test_get_gate(self):
        """Test getting gate from action index."""
        library = GateLibrary(qumode_dim=5, n_discrete_params=3)
        
        for idx in range(min(10, library.n_actions)):
            gate = library.get_gate(idx)
            assert isinstance(gate, Gate)
            assert gate.qumode_dim == library.qumode_dim
    
    def test_sample_random_action(self):
        """Test random action sampling."""
        library = GateLibrary(qumode_dim=5)
        
        for _ in range(10):
            action = library.sample_random_action()
            assert 0 <= action < library.n_actions
    
    def test_sample_random_gate(self):
        """Test random gate sampling."""
        library = GateLibrary(qumode_dim=5)
        
        for _ in range(10):
            gate = library.sample_random_gate()
            assert isinstance(gate, Gate)
    
    def test_library_len(self):
        """Test library length."""
        library = GateLibrary(qumode_dim=5)
        assert len(library) == library.n_actions


class TestGateSequence:
    """Test suite for GateSequence."""
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        seq = GateSequence()
        
        assert len(seq) == 0
        assert seq.depth() == 0
    
    def test_append_gate(self):
        """Test appending gates."""
        seq = GateSequence()
        gate = Gate(GateType.RX, [np.pi/4], qumode_dim=5)
        
        seq.append(gate)
        
        assert len(seq) == 1
        assert seq.gates[0] is gate
    
    def test_append_chaining(self):
        """Test that append returns self for chaining."""
        seq = GateSequence()
        g1 = Gate(GateType.RX, [np.pi/4], qumode_dim=5)
        g2 = Gate(GateType.RY, [np.pi/3], qumode_dim=5)
        
        seq.append(g1).append(g2)
        
        assert len(seq) == 2
    
    def test_sequence_to_operator(self):
        """Test converting sequence to operator."""
        qumode_dim = 5
        seq = GateSequence()
        
        g1 = Gate(GateType.RX, [np.pi/4], qumode_dim)
        g2 = Gate(GateType.RY, [np.pi/3], qumode_dim)
        
        seq.append(g1).append(g2)
        
        U = seq.to_operator()
        
        # Should equal U2 @ U1
        expected = g2.to_operator() @ g1.to_operator()
        np.testing.assert_array_almost_equal(U.matrix, expected.matrix)
    
    def test_empty_sequence_to_operator_raises(self):
        """Test that empty sequence raises on to_operator."""
        seq = GateSequence()
        
        with pytest.raises(ValueError):
            seq.to_operator()
    
    def test_sequence_copy(self):
        """Test sequence copy."""
        seq = GateSequence()
        g1 = Gate(GateType.RX, [np.pi/4], qumode_dim=5)
        seq.append(g1)
        
        seq_copy = seq.copy()
        
        assert len(seq_copy) == 1
        assert seq_copy is not seq
        assert seq_copy.gates is not seq.gates
    
    def test_sequence_unitary(self):
        """Test that gate sequence produces unitary operator."""
        qumode_dim = 5
        seq = GateSequence()
        
        # Add several gates
        seq.append(Gate(GateType.RX, [np.pi/4], qumode_dim))
        seq.append(Gate(GateType.DISPLACEMENT, [0.3, 0.0], qumode_dim))
        seq.append(Gate(GateType.JAYNES_CUMMINGS, [0.5], qumode_dim))
        seq.append(Gate(GateType.RZ, [np.pi/6], qumode_dim))
        
        U = seq.to_operator()
        
        # U†U = I
        product = U.dag() @ U
        np.testing.assert_array_almost_equal(
            product.matrix,
            np.eye(2 * qumode_dim, dtype=complex),
            decimal=5
        )
