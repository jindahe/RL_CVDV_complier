"""
Tests for Hamiltonian module.
"""

import numpy as np
import pytest
from hybrid_compiler.hamiltonian import (
    Hamiltonian, HamiltonianBuilder, fidelity, process_fidelity, operator_distance
)
from hybrid_compiler.operators import (
    Operator, pauli_z, qumode_identity, qubit_identity, tensor, number_op
)


class TestHamiltonian:
    """Test suite for Hamiltonian class."""
    
    def test_empty_hamiltonian(self):
        """Test empty Hamiltonian initialization."""
        H = Hamiltonian(qumode_dim=5, n_qubits=1, n_qumodes=1)
        
        assert H.dim == 10  # 2 * 5
        assert len(H.terms) == 0
        
        # Empty Hamiltonian should be zero matrix
        expected = np.zeros((10, 10), dtype=complex)
        np.testing.assert_array_almost_equal(H.matrix, expected)
    
    def test_add_term(self):
        """Test adding terms to Hamiltonian."""
        H = Hamiltonian(qumode_dim=5, n_qubits=1, n_qumodes=1)
        
        # Add σz ⊗ I
        sz_I = tensor(pauli_z(), qumode_identity(5))
        H.add_term(1.0, sz_I)
        
        assert len(H.terms) == 1
        np.testing.assert_array_almost_equal(H.matrix, sz_I.matrix)
    
    def test_add_term_chaining(self):
        """Test that add_term returns self for chaining."""
        H = Hamiltonian(qumode_dim=5, n_qubits=1, n_qumodes=1)
        
        sz_I = tensor(pauli_z(), qumode_identity(5))
        I_n = tensor(qubit_identity(), number_op(5))
        
        result = H.add_term(1.0, sz_I).add_term(2.0, I_n)
        
        assert result is H
        assert len(H.terms) == 2
    
    def test_hamiltonian_hermiticity(self):
        """Test that constructed Hamiltonians are Hermitian."""
        builder = HamiltonianBuilder(qumode_dim=5)
        
        # All standard models should give Hermitian Hamiltonians
        models = [
            builder.rabi_model(1.0, 1.0, 0.1),
            builder.jaynes_cummings_model(1.0, 1.0, 0.1),
            builder.dispersive_model(1.0, 1.0, 0.05),
            builder.kerr_oscillator(1.0, 0.01),
        ]
        
        for H in models:
            np.testing.assert_array_almost_equal(
                H.matrix, H.matrix.conj().T,
                err_msg=f"Hamiltonian should be Hermitian"
            )
    
    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        H = Hamiltonian(qumode_dim=5, n_qubits=1, n_qumodes=1)
        
        # Wrong dimension operator
        wrong_op = Operator(np.eye(3), "wrong")
        
        with pytest.raises(ValueError):
            H.add_term(1.0, wrong_op)
    
    def test_eigendecomposition(self):
        """Test eigenvalue computation."""
        # Simple test: σz ⊗ I should have eigenvalues ±1 (each 5-fold degenerate)
        H = Hamiltonian(qumode_dim=5, n_qubits=1, n_qumodes=1)
        sz_I = tensor(pauli_z(), qumode_identity(5))
        H.add_term(1.0, sz_I)
        
        eigenvalues, _ = H.eigendecomposition()
        
        # Should have 5 eigenvalues of -1 and 5 eigenvalues of +1
        assert np.sum(np.isclose(eigenvalues, -1)) == 5
        assert np.sum(np.isclose(eigenvalues, 1)) == 5
    
    def test_ground_state(self):
        """Test ground state computation."""
        H = Hamiltonian(qumode_dim=5, n_qubits=1, n_qumodes=1)
        sz_I = tensor(pauli_z(), qumode_identity(5))
        H.add_term(1.0, sz_I)
        
        energy, state = H.ground_state()
        
        assert np.isclose(energy, -1.0)
        assert np.isclose(np.linalg.norm(state), 1.0)  # Normalized
    
    def test_time_evolution_unitary(self):
        """Test that time evolution is unitary."""
        builder = HamiltonianBuilder(qumode_dim=5)
        H = builder.jaynes_cummings_model(1.0, 1.0, 0.1)
        
        U = H.time_evolution(t=1.0)
        
        # U†U = I
        product = U.dag() @ U
        np.testing.assert_array_almost_equal(
            product.matrix, np.eye(H.dim, dtype=complex), decimal=10
        )


class TestHamiltonianBuilder:
    """Test suite for HamiltonianBuilder."""
    
    def test_rabi_model(self):
        """Test Rabi model construction."""
        builder = HamiltonianBuilder(qumode_dim=5)
        H = builder.rabi_model(omega_q=1.0, omega_c=2.0, g=0.1)
        
        assert H.dim == 10
        assert len(H.terms) == 3  # Qubit, cavity, coupling
    
    def test_jaynes_cummings_model(self):
        """Test Jaynes-Cummings model construction."""
        builder = HamiltonianBuilder(qumode_dim=5)
        H = builder.jaynes_cummings_model(omega_q=1.0, omega_c=2.0, g=0.1)
        
        assert H.dim == 10
        assert len(H.terms) == 3
    
    def test_dispersive_model(self):
        """Test dispersive model construction."""
        builder = HamiltonianBuilder(qumode_dim=5)
        H = builder.dispersive_model(omega_q=1.0, omega_c=2.0, chi=0.05)
        
        assert H.dim == 10
        assert len(H.terms) == 3
    
    def test_kerr_oscillator(self):
        """Test Kerr oscillator construction."""
        builder = HamiltonianBuilder(qumode_dim=5)
        H = builder.kerr_oscillator(omega=1.0, K=0.01)
        
        assert H.dim == 10
        assert len(H.terms) == 2  # Linear and Kerr terms
    
    def test_custom_hamiltonian(self):
        """Test custom Hamiltonian construction."""
        builder = HamiltonianBuilder(qumode_dim=5)
        
        H = builder.custom([
            (1.0, "Iz*n"),
            (0.5, "Ix*q"),
        ])
        
        assert H.dim == 10
        assert len(H.terms) == 2


class TestFidelityFunctions:
    """Test suite for fidelity functions."""
    
    def test_fidelity_identity(self):
        """Test fidelity between identical operators is 1."""
        U = Operator(np.eye(4, dtype=complex), "I")
        
        f = fidelity(U, U)
        assert np.isclose(f, 1.0)
    
    def test_fidelity_orthogonal(self):
        """Test fidelity between orthogonal unitaries is 0."""
        # σx and σy are orthogonal in trace sense
        from hybrid_compiler.operators import pauli_x, pauli_y
        
        X = pauli_x()
        Y = pauli_y()
        
        f = fidelity(X, Y)
        assert np.isclose(f, 0.0)
    
    def test_process_fidelity(self):
        """Test process fidelity."""
        U = Operator(np.eye(4, dtype=complex), "I")
        
        f = process_fidelity(U, U)
        assert np.isclose(f, 1.0)
    
    def test_operator_distance_zero(self):
        """Test distance between identical operators is zero."""
        U = Operator(np.eye(4, dtype=complex), "I")
        
        d = operator_distance(U, U)
        assert np.isclose(d, 0.0)
    
    def test_operator_distance_positive(self):
        """Test distance between different operators is positive."""
        from hybrid_compiler.operators import pauli_x, pauli_z
        
        X = pauli_x()
        Z = pauli_z()
        
        d = operator_distance(X, Z)
        assert d > 0
    
    def test_operator_distance_hamiltonian(self):
        """Test distance works with Hamiltonian objects."""
        builder = HamiltonianBuilder(qumode_dim=5)
        
        H1 = builder.rabi_model(1.0, 1.0, 0.1)
        H2 = builder.rabi_model(1.0, 1.0, 0.2)
        
        d = operator_distance(H1, H2)
        assert d > 0
