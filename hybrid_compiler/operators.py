"""
Quantum operators for hybrid CV-DV (qubit-qumode) systems.

This module provides operator representations for:
- Discrete-variable (DV) qubit operators: Pauli X, Y, Z, identity
- Continuous-variable (CV) qumode operators: creation, annihilation, number, position, momentum
- Hybrid interaction operators for qubit-qumode coupling
"""

import numpy as np
from typing import Union, Optional
from scipy.linalg import expm


class Operator:
    """Base class for quantum operators."""
    
    def __init__(self, matrix: np.ndarray, name: str = ""):
        """
        Initialize an operator.
        
        Args:
            matrix: Matrix representation of the operator
            name: Human-readable name
        """
        self.matrix = np.array(matrix, dtype=complex)
        self.name = name
        self.dim = matrix.shape[0]
    
    def __repr__(self) -> str:
        return f"Operator({self.name}, dim={self.dim})"
    
    def __matmul__(self, other: "Operator") -> "Operator":
        """Matrix multiplication."""
        if isinstance(other, Operator):
            return Operator(self.matrix @ other.matrix, f"({self.name}@{other.name})")
        return NotImplemented
    
    def __add__(self, other: "Operator") -> "Operator":
        """Addition."""
        if isinstance(other, Operator):
            return Operator(self.matrix + other.matrix, f"({self.name}+{other.name})")
        return NotImplemented
    
    def __sub__(self, other: "Operator") -> "Operator":
        """Subtraction."""
        if isinstance(other, Operator):
            return Operator(self.matrix - other.matrix, f"({self.name}-{other.name})")
        return NotImplemented
    
    def __mul__(self, scalar: complex) -> "Operator":
        """Scalar multiplication."""
        return Operator(scalar * self.matrix, f"{scalar}*{self.name}")
    
    def __rmul__(self, scalar: complex) -> "Operator":
        """Scalar multiplication from left."""
        return self.__mul__(scalar)
    
    def dag(self) -> "Operator":
        """Return the Hermitian conjugate (dagger)."""
        return Operator(self.matrix.conj().T, f"{self.name}†")
    
    def trace(self) -> complex:
        """Compute the trace."""
        return np.trace(self.matrix)
    
    def norm(self) -> float:
        """Compute the Frobenius norm."""
        return np.linalg.norm(self.matrix, 'fro')
    
    def exp(self, coeff: complex = 1.0) -> "Operator":
        """
        Compute matrix exponential exp(coeff * self).
        
        Args:
            coeff: Coefficient in the exponent
        
        Returns:
            exp(coeff * operator)
        """
        return Operator(expm(coeff * self.matrix), f"exp({coeff}*{self.name})")


# =============================================================================
# Qubit (DV) Operators
# =============================================================================

def pauli_x() -> Operator:
    """Pauli X operator."""
    return Operator(np.array([[0, 1], [1, 0]]), "σx")


def pauli_y() -> Operator:
    """Pauli Y operator."""
    return Operator(np.array([[0, -1j], [1j, 0]]), "σy")


def pauli_z() -> Operator:
    """Pauli Z operator."""
    return Operator(np.array([[1, 0], [0, -1]]), "σz")


def qubit_identity() -> Operator:
    """Qubit identity operator."""
    return Operator(np.eye(2), "I₂")


def sigma_plus() -> Operator:
    """Raising operator σ+ = |1⟩⟨0|."""
    return Operator(np.array([[0, 1], [0, 0]]), "σ+")


def sigma_minus() -> Operator:
    """Lowering operator σ- = |0⟩⟨1|."""
    return Operator(np.array([[0, 0], [1, 0]]), "σ-")


def hadamard() -> Operator:
    """Hadamard gate."""
    return Operator(np.array([[1, 1], [1, -1]]) / np.sqrt(2), "H")


def rotation_x(theta: float) -> Operator:
    """Rotation around X axis by angle theta."""
    return Operator(
        np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ]),
        f"Rx({theta:.3f})"
    )


def rotation_y(theta: float) -> Operator:
    """Rotation around Y axis by angle theta."""
    return Operator(
        np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ]),
        f"Ry({theta:.3f})"
    )


def rotation_z(theta: float) -> Operator:
    """Rotation around Z axis by angle theta."""
    return Operator(
        np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ]),
        f"Rz({theta:.3f})"
    )


# =============================================================================
# Qumode (CV) Operators
# =============================================================================

def creation(dim: int) -> Operator:
    """
    Creation (raising) operator a†.
    
    Args:
        dim: Fock space truncation dimension
    """
    matrix = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        matrix[n + 1, n] = np.sqrt(n + 1)
    return Operator(matrix, "a†")


def annihilation(dim: int) -> Operator:
    """
    Annihilation (lowering) operator a.
    
    Args:
        dim: Fock space truncation dimension
    """
    matrix = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        matrix[n, n + 1] = np.sqrt(n + 1)
    return Operator(matrix, "a")


def number_op(dim: int) -> Operator:
    """
    Number operator n = a†a.
    
    Args:
        dim: Fock space truncation dimension
    """
    matrix = np.diag(np.arange(dim, dtype=complex))
    return Operator(matrix, "n")


def position(dim: int) -> Operator:
    """
    Position quadrature operator q = (a + a†) / √2.
    
    Args:
        dim: Fock space truncation dimension
    """
    a = annihilation(dim)
    a_dag = creation(dim)
    q = (a + a_dag) * (1.0 / np.sqrt(2))
    q.name = "q"
    return q


def momentum(dim: int) -> Operator:
    """
    Momentum quadrature operator p = i(a† - a) / √2.
    
    Args:
        dim: Fock space truncation dimension
    """
    a = annihilation(dim)
    a_dag = creation(dim)
    p = (a_dag - a) * (1j / np.sqrt(2))
    p.name = "p"
    return p


def qumode_identity(dim: int) -> Operator:
    """Qumode identity operator."""
    return Operator(np.eye(dim), f"I_{dim}")


def displacement(dim: int, alpha: complex) -> Operator:
    """
    Displacement operator D(α) = exp(α*a† - α*a).
    
    Args:
        dim: Fock space truncation dimension
        alpha: Displacement amplitude
    """
    a = annihilation(dim)
    a_dag = creation(dim)
    gen = alpha * a_dag - np.conj(alpha) * a
    D = gen.exp(1.0)
    D.name = f"D({alpha:.3f})"
    return D


def squeeze(dim: int, r: float, phi: float = 0.0) -> Operator:
    """
    Squeezing operator S(ξ) = exp((ξ*a†² - ξ*a²)/2) where ξ = r*exp(i*phi).
    
    Args:
        dim: Fock space truncation dimension
        r: Squeezing magnitude
        phi: Squeezing phase
    """
    a = annihilation(dim)
    a_dag = creation(dim)
    xi = r * np.exp(1j * phi)
    gen = (np.conj(xi) * (a @ a) - xi * (a_dag @ a_dag)) * 0.5
    S = gen.exp(1.0)
    S.name = f"S({r:.3f},{phi:.3f})"
    return S


def rotation_mode(dim: int, theta: float) -> Operator:
    """
    Phase rotation operator R(θ) = exp(i*θ*n).
    
    Args:
        dim: Fock space truncation dimension
        theta: Rotation angle
    """
    n = number_op(dim)
    R = n.exp(1j * theta)
    R.name = f"R({theta:.3f})"
    return R


# =============================================================================
# Tensor Product Utilities
# =============================================================================

def tensor(*operators: Operator) -> Operator:
    """
    Compute tensor product of operators.
    
    Args:
        operators: Variable number of operators to tensor
    
    Returns:
        Tensor product operator
    """
    if len(operators) == 0:
        raise ValueError("Need at least one operator")
    
    result = operators[0].matrix
    name = operators[0].name
    
    for op in operators[1:]:
        result = np.kron(result, op.matrix)
        name = f"{name}⊗{op.name}"
    
    return Operator(result, name)


def embed_qubit(qubit_op: Operator, qumode_dim: int, on_qubit: bool = True) -> Operator:
    """
    Embed a qubit operator in the hybrid space.
    
    Args:
        qubit_op: Qubit operator (2x2)
        qumode_dim: Dimension of qumode space
        on_qubit: If True, tensor with qumode identity; if False, identity on qubit
    
    Returns:
        Operator in the hybrid space
    """
    if on_qubit:
        return tensor(qubit_op, qumode_identity(qumode_dim))
    else:
        return tensor(qubit_identity(), qubit_op)


def embed_qumode(qumode_op: Operator, on_qumode: bool = True) -> Operator:
    """
    Embed a qumode operator in the hybrid space.
    
    Args:
        qumode_op: Qumode operator
        on_qumode: If True, tensor with qubit identity; if False, identity on qumode
    
    Returns:
        Operator in the hybrid space
    """
    if on_qumode:
        return tensor(qubit_identity(), qumode_op)
    else:
        return tensor(qumode_op, qubit_identity())


# =============================================================================
# Hybrid Interaction Operators
# =============================================================================

def jaynes_cummings(qumode_dim: int, g: float = 1.0) -> Operator:
    """
    Jaynes-Cummings interaction: H_JC = g(σ+a + σ-a†).
    
    This is the fundamental qubit-qumode coupling in cavity QED.
    
    Args:
        qumode_dim: Fock space truncation dimension
        g: Coupling strength
    
    Returns:
        Jaynes-Cummings Hamiltonian
    """
    a = annihilation(qumode_dim)
    a_dag = creation(qumode_dim)
    sp = sigma_plus()
    sm = sigma_minus()
    
    H = g * (tensor(sp, a) + tensor(sm, a_dag))
    H.name = f"H_JC(g={g})"
    return H


def anti_jaynes_cummings(qumode_dim: int, g: float = 1.0) -> Operator:
    """
    Anti-Jaynes-Cummings interaction: H_AJC = g(σ+a† + σ-a).
    
    Args:
        qumode_dim: Fock space truncation dimension
        g: Coupling strength
    
    Returns:
        Anti-Jaynes-Cummings Hamiltonian
    """
    a = annihilation(qumode_dim)
    a_dag = creation(qumode_dim)
    sp = sigma_plus()
    sm = sigma_minus()
    
    H = g * (tensor(sp, a_dag) + tensor(sm, a))
    H.name = f"H_AJC(g={g})"
    return H


def dispersive_coupling(qumode_dim: int, chi: float = 1.0) -> Operator:
    """
    Dispersive coupling: H_disp = χ * σz ⊗ n.
    
    Args:
        qumode_dim: Fock space truncation dimension
        chi: Coupling strength
    
    Returns:
        Dispersive coupling Hamiltonian
    """
    sz = pauli_z()
    n = number_op(qumode_dim)
    
    H = chi * tensor(sz, n)
    H.name = f"H_disp(χ={chi})"
    return H


def qubit_position_coupling(qumode_dim: int, g: float = 1.0) -> Operator:
    """
    Qubit-position coupling: H = g * σz ⊗ q.
    
    Args:
        qumode_dim: Fock space truncation dimension
        g: Coupling strength
    
    Returns:
        Qubit-position coupling Hamiltonian
    """
    sz = pauli_z()
    q = position(qumode_dim)
    
    H = g * tensor(sz, q)
    H.name = f"H_zq(g={g})"
    return H


def conditional_displacement(qumode_dim: int, alpha: complex) -> Operator:
    """
    Conditional displacement: |0⟩⟨0| ⊗ D(α) + |1⟩⟨1| ⊗ D(-α).
    
    This gate displaces the qumode conditioned on the qubit state.
    
    Args:
        qumode_dim: Fock space truncation dimension
        alpha: Displacement amplitude
    
    Returns:
        Conditional displacement operator
    """
    proj0 = Operator(np.array([[1, 0], [0, 0]]), "|0⟩⟨0|")
    proj1 = Operator(np.array([[0, 0], [0, 1]]), "|1⟩⟨1|")
    
    D_plus = displacement(qumode_dim, alpha)
    D_minus = displacement(qumode_dim, -alpha)
    
    CD = tensor(proj0, D_plus) + tensor(proj1, D_minus)
    CD.name = f"CD({alpha:.3f})"
    return CD
