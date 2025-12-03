"""
Tests for quantum operators module.
"""

import numpy as np
import pytest
from hybrid_compiler.operators import (
    Operator, pauli_x, pauli_y, pauli_z, qubit_identity,
    sigma_plus, sigma_minus, hadamard,
    rotation_x, rotation_y, rotation_z,
    creation, annihilation, number_op, position, momentum, qumode_identity,
    displacement, squeeze, rotation_mode,
    tensor, embed_qubit, embed_qumode,
    jaynes_cummings, anti_jaynes_cummings, dispersive_coupling,
    conditional_displacement
)


class TestQubitOperators:
    """Test suite for qubit operators."""
    
    def test_pauli_matrices_hermitian(self):
        """Pauli matrices should be Hermitian."""
        for op in [pauli_x(), pauli_y(), pauli_z()]:
            np.testing.assert_array_almost_equal(op.matrix, op.dag().matrix)
    
    def test_pauli_matrices_square_to_identity(self):
        """Pauli matrices squared should equal identity."""
        I = qubit_identity()
        for op in [pauli_x(), pauli_y(), pauli_z()]:
            result = op @ op
            np.testing.assert_array_almost_equal(result.matrix, I.matrix)
    
    def test_pauli_anticommutation(self):
        """Pauli matrices should anticommute."""
        X, Y, Z = pauli_x(), pauli_y(), pauli_z()
        
        # XY + YX = 0
        result = (X @ Y) + (Y @ X)
        np.testing.assert_array_almost_equal(result.matrix, np.zeros((2, 2)))
    
    def test_sigma_plus_minus(self):
        """Test raising and lowering operators."""
        sp = sigma_plus()
        sm = sigma_minus()
        
        # σ+ @ σ- = |0⟩⟨0| (projector onto ground state)
        proj0 = sp @ sm
        np.testing.assert_array_almost_equal(
            proj0.matrix, np.array([[1, 0], [0, 0]])
        )
        
        # σ- @ σ+ = |1⟩⟨1| (projector onto excited state)
        proj1 = sm @ sp
        np.testing.assert_array_almost_equal(
            proj1.matrix, np.array([[0, 0], [0, 1]])
        )
    
    def test_hadamard_properties(self):
        """Test Hadamard gate properties."""
        H = hadamard()
        I = qubit_identity()
        
        # H is Hermitian
        np.testing.assert_array_almost_equal(H.matrix, H.dag().matrix)
        
        # H² = I
        result = H @ H
        np.testing.assert_array_almost_equal(result.matrix, I.matrix)
    
    def test_rotation_gates(self):
        """Test rotation gates."""
        # Rx(π) should give -i*X
        rx_pi = rotation_x(np.pi)
        expected = -1j * pauli_x().matrix
        np.testing.assert_array_almost_equal(rx_pi.matrix, expected)
        
        # Ry(π) should give -i*Y
        ry_pi = rotation_y(np.pi)
        expected = -1j * pauli_y().matrix
        np.testing.assert_array_almost_equal(ry_pi.matrix, expected)
        
        # Rz(2π) should give -I (global phase)
        rz_2pi = rotation_z(2 * np.pi)
        expected = -np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(rz_2pi.matrix, expected)


class TestQumodeOperators:
    """Test suite for qumode (continuous-variable) operators."""
    
    def test_creation_annihilation_commutator(self):
        """Test [a, a†] = 1."""
        dim = 10
        a = annihilation(dim)
        a_dag = creation(dim)
        
        commutator = (a @ a_dag) - (a_dag @ a)
        # Should be close to identity (truncation effects near edges)
        np.testing.assert_array_almost_equal(
            commutator.matrix[:dim-1, :dim-1],
            np.eye(dim-1, dtype=complex),
            decimal=10
        )
    
    def test_number_operator(self):
        """Test number operator n = a†a."""
        dim = 10
        a = annihilation(dim)
        a_dag = creation(dim)
        n = number_op(dim)
        
        # n = a†a
        n_computed = a_dag @ a
        np.testing.assert_array_almost_equal(n_computed.matrix, n.matrix)
        
        # Eigenvalues should be 0, 1, 2, ...
        eigenvalues = np.diag(n.matrix)
        np.testing.assert_array_almost_equal(eigenvalues, np.arange(dim))
    
    def test_position_momentum_commutator(self):
        """Test [q, p] = i."""
        dim = 20
        q = position(dim)
        p = momentum(dim)
        
        commutator = (q @ p) - (p @ q)
        # Should be close to i*I (truncation effects near edges)
        expected = 1j * np.eye(dim, dtype=complex)
        np.testing.assert_array_almost_equal(
            commutator.matrix[:dim-2, :dim-2],
            expected[:dim-2, :dim-2],
            decimal=5
        )
    
    def test_displacement_operator(self):
        """Test displacement operator properties."""
        dim = 20
        alpha = 0.5 + 0.3j
        
        D = displacement(dim, alpha)
        D_dag = D.dag()
        
        # D should be unitary: D†D = I
        product = D_dag @ D
        np.testing.assert_array_almost_equal(
            product.matrix, np.eye(dim, dtype=complex), decimal=5
        )
    
    def test_squeezing_operator(self):
        """Test squeezing operator is unitary."""
        dim = 20
        r = 0.3
        
        S = squeeze(dim, r)
        S_dag = S.dag()
        
        # S should be unitary
        product = S_dag @ S
        np.testing.assert_array_almost_equal(
            product.matrix, np.eye(dim, dtype=complex), decimal=5
        )
    
    def test_phase_rotation(self):
        """Test phase rotation operator."""
        dim = 10
        theta = np.pi / 3
        
        R = rotation_mode(dim, theta)
        
        # R should be diagonal
        assert np.allclose(R.matrix, np.diag(np.diag(R.matrix)))
        
        # Diagonal elements should be exp(i*n*theta)
        expected_diag = np.exp(1j * theta * np.arange(dim))
        np.testing.assert_array_almost_equal(np.diag(R.matrix), expected_diag)


class TestTensorProduct:
    """Test suite for tensor product operations."""
    
    def test_tensor_dimension(self):
        """Test tensor product gives correct dimension."""
        op1 = pauli_x()  # 2x2
        op2 = qumode_identity(5)  # 5x5
        
        result = tensor(op1, op2)
        assert result.dim == 10
        assert result.matrix.shape == (10, 10)
    
    def test_tensor_associativity(self):
        """Test tensor product is associative."""
        A = pauli_x()
        B = pauli_y()
        C = pauli_z()
        
        # (A ⊗ B) ⊗ C should equal A ⊗ (B ⊗ C)
        left = tensor(tensor(A, B), C)
        right = tensor(A, tensor(B, C))
        
        np.testing.assert_array_almost_equal(left.matrix, right.matrix)
    
    def test_embed_qubit(self):
        """Test embedding qubit operator in hybrid space."""
        dim = 5
        X = pauli_x()
        
        embedded = embed_qubit(X, dim)
        
        # Should be 10x10 (2*5)
        assert embedded.dim == 10
        
        # Should equal X ⊗ I_5
        expected = tensor(X, qumode_identity(dim))
        np.testing.assert_array_almost_equal(embedded.matrix, expected.matrix)
    
    def test_embed_qumode(self):
        """Test embedding qumode operator in hybrid space."""
        dim = 5
        n = number_op(dim)
        
        embedded = embed_qumode(n)
        
        # Should be 10x10 (2*5)
        assert embedded.dim == 10
        
        # Should equal I_2 ⊗ n
        expected = tensor(qubit_identity(), n)
        np.testing.assert_array_almost_equal(embedded.matrix, expected.matrix)


class TestHybridOperators:
    """Test suite for hybrid qubit-qumode operators."""
    
    def test_jaynes_cummings_hermitian(self):
        """Jaynes-Cummings Hamiltonian should be Hermitian."""
        dim = 10
        H_JC = jaynes_cummings(dim, g=1.0)
        
        np.testing.assert_array_almost_equal(H_JC.matrix, H_JC.dag().matrix)
    
    def test_dispersive_coupling_hermitian(self):
        """Dispersive coupling should be Hermitian."""
        dim = 10
        H_disp = dispersive_coupling(dim, chi=0.5)
        
        np.testing.assert_array_almost_equal(H_disp.matrix, H_disp.dag().matrix)
    
    def test_conditional_displacement_unitary(self):
        """Conditional displacement should be unitary."""
        dim = 15
        alpha = 0.3
        
        CD = conditional_displacement(dim, alpha)
        
        # CD†CD = I
        product = CD.dag() @ CD
        np.testing.assert_array_almost_equal(
            product.matrix, np.eye(2 * dim, dtype=complex), decimal=5
        )


class TestOperatorArithmetic:
    """Test operator arithmetic operations."""
    
    def test_addition(self):
        """Test operator addition."""
        X = pauli_x()
        Z = pauli_z()
        
        result = X + Z
        expected = X.matrix + Z.matrix
        np.testing.assert_array_almost_equal(result.matrix, expected)
    
    def test_subtraction(self):
        """Test operator subtraction."""
        X = pauli_x()
        Y = pauli_y()
        
        result = X - Y
        expected = X.matrix - Y.matrix
        np.testing.assert_array_almost_equal(result.matrix, expected)
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        X = pauli_x()
        
        result1 = 2.0 * X
        result2 = X * 2.0
        
        expected = 2.0 * X.matrix
        np.testing.assert_array_almost_equal(result1.matrix, expected)
        np.testing.assert_array_almost_equal(result2.matrix, expected)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        X = pauli_x()
        Y = pauli_y()
        
        result = X @ Y
        expected = X.matrix @ Y.matrix
        np.testing.assert_array_almost_equal(result.matrix, expected)
    
    def test_trace(self):
        """Test trace computation."""
        I = qubit_identity()
        assert I.trace() == 2
        
        Z = pauli_z()
        assert Z.trace() == 0
    
    def test_norm(self):
        """Test Frobenius norm."""
        I = qubit_identity()
        assert np.isclose(I.norm(), np.sqrt(2))
    
    def test_matrix_exponential(self):
        """Test matrix exponential."""
        Z = pauli_z()
        
        # exp(i*π*Z/2) should give diag(i, -i)
        result = Z.exp(1j * np.pi / 2)
        expected = np.diag([1j, -1j])
        np.testing.assert_array_almost_equal(result.matrix, expected)
