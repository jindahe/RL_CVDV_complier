"""
Hamiltonian representation for hybrid CV-DV quantum systems.

This module provides tools for constructing and manipulating Hamiltonians
that describe hybrid qubit-qumode systems.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .operators import (
    Operator, tensor, pauli_x, pauli_y, pauli_z, qubit_identity,
    creation, annihilation, number_op, position, momentum, qumode_identity,
    jaynes_cummings, dispersive_coupling
)


class Hamiltonian:
    """
    Represents a Hamiltonian for a hybrid qubit-qumode system.
    
    The Hamiltonian can be constructed from a sum of terms, each with
    a coefficient and an operator.
    """
    
    def __init__(self, qumode_dim: int = 10, n_qubits: int = 1, n_qumodes: int = 1):
        """
        Initialize a Hamiltonian for a hybrid system.
        
        Args:
            qumode_dim: Truncation dimension for each qumode's Fock space
            n_qubits: Number of qubits
            n_qumodes: Number of qumodes
        """
        self.qumode_dim = qumode_dim
        self.n_qubits = n_qubits
        self.n_qumodes = n_qumodes
        
        # Total Hilbert space dimension: 2^n_qubits * qumode_dim^n_qumodes
        self.dim = (2 ** n_qubits) * (qumode_dim ** n_qumodes)
        
        # Terms: list of (coefficient, operator) pairs
        self.terms: List[Tuple[complex, Operator]] = []
        
        # Cached matrix representation
        self._matrix: Optional[np.ndarray] = None
    
    def add_term(self, coeff: complex, operator: Operator) -> "Hamiltonian":
        """
        Add a term to the Hamiltonian.
        
        Args:
            coeff: Coefficient for the term
            operator: Operator for the term (must have correct dimension)
        
        Returns:
            self for chaining
        """
        if operator.dim != self.dim:
            raise ValueError(f"Operator dimension {operator.dim} != system dimension {self.dim}")
        
        self.terms.append((coeff, operator))
        self._matrix = None  # Invalidate cache
        return self
    
    @property
    def matrix(self) -> np.ndarray:
        """Get the matrix representation of the Hamiltonian."""
        if self._matrix is None:
            self._matrix = np.zeros((self.dim, self.dim), dtype=complex)
            for coeff, op in self.terms:
                self._matrix += coeff * op.matrix
        return self._matrix
    
    def to_operator(self) -> Operator:
        """Convert to an Operator object."""
        return Operator(self.matrix, "H")
    
    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.
        
        Returns:
            eigenvalues: Array of eigenvalues (sorted)
            eigenvectors: Array of eigenvectors as columns
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        return eigenvalues, eigenvectors
    
    def ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Get the ground state energy and state vector.
        
        Returns:
            energy: Ground state energy
            state: Ground state vector
        """
        eigenvalues, eigenvectors = self.eigendecomposition()
        return eigenvalues[0], eigenvectors[:, 0]
    
    def time_evolution(self, t: float) -> Operator:
        """
        Compute the time evolution operator U(t) = exp(-i*H*t).
        
        Args:
            t: Time
        
        Returns:
            Unitary time evolution operator
        """
        return Operator(self.matrix, "H").exp(-1j * t)
    
    def __repr__(self) -> str:
        return f"Hamiltonian(dim={self.dim}, terms={len(self.terms)})"


class HamiltonianBuilder:
    """
    Helper class for building common hybrid Hamiltonians.
    
    This provides convenient methods for constructing standard
    Hamiltonians found in cavity QED, circuit QED, and hybrid systems.
    """
    
    def __init__(self, qumode_dim: int = 10):
        """
        Initialize the builder.
        
        Args:
            qumode_dim: Default Fock space truncation dimension
        """
        self.qumode_dim = qumode_dim
    
    def single_qubit_qumode(self) -> Hamiltonian:
        """Create an empty Hamiltonian for a single qubit and single qumode."""
        return Hamiltonian(self.qumode_dim, n_qubits=1, n_qumodes=1)
    
    def rabi_model(self, omega_q: float, omega_c: float, g: float) -> Hamiltonian:
        """
        Create the Rabi model Hamiltonian.
        
        H = (ω_q/2)σz + ω_c*a†a + g*σx(a + a†)
        
        This is the fundamental model of light-matter interaction.
        
        Args:
            omega_q: Qubit frequency
            omega_c: Cavity (qumode) frequency  
            g: Coupling strength
        
        Returns:
            Rabi model Hamiltonian
        """
        H = self.single_qubit_qumode()
        dim = self.qumode_dim
        
        # Qubit term: (ω_q/2)σz ⊗ I_cavity
        sz = pauli_z()
        I_c = qumode_identity(dim)
        H.add_term(omega_q / 2, tensor(sz, I_c))
        
        # Cavity term: I_qubit ⊗ ω_c*n
        I_q = qubit_identity()
        n = number_op(dim)
        H.add_term(omega_c, tensor(I_q, n))
        
        # Coupling: g*σx ⊗ (a + a†)
        sx = pauli_x()
        a = annihilation(dim)
        a_dag = creation(dim)
        H.add_term(g, tensor(sx, a + a_dag))
        
        return H
    
    def jaynes_cummings_model(self, omega_q: float, omega_c: float, g: float) -> Hamiltonian:
        """
        Create the Jaynes-Cummings model Hamiltonian.
        
        H = (ω_q/2)σz + ω_c*a†a + g(σ+a + σ-a†)
        
        This is the rotating wave approximation of the Rabi model.
        
        Args:
            omega_q: Qubit frequency
            omega_c: Cavity frequency
            g: Coupling strength
        
        Returns:
            Jaynes-Cummings Hamiltonian
        """
        H = self.single_qubit_qumode()
        dim = self.qumode_dim
        
        # Qubit term
        sz = pauli_z()
        I_c = qumode_identity(dim)
        H.add_term(omega_q / 2, tensor(sz, I_c))
        
        # Cavity term
        I_q = qubit_identity()
        n = number_op(dim)
        H.add_term(omega_c, tensor(I_q, n))
        
        # JC coupling
        H.add_term(1.0, jaynes_cummings(dim, g))
        
        return H
    
    def dispersive_model(self, omega_q: float, omega_c: float, chi: float) -> Hamiltonian:
        """
        Create a dispersive Hamiltonian.
        
        H = (ω_q/2)σz + ω_c*a†a + χ*σz*n
        
        This describes the dispersive regime of circuit QED.
        
        Args:
            omega_q: Qubit frequency
            omega_c: Cavity frequency
            chi: Dispersive shift
        
        Returns:
            Dispersive Hamiltonian
        """
        H = self.single_qubit_qumode()
        dim = self.qumode_dim
        
        # Qubit term
        sz = pauli_z()
        I_c = qumode_identity(dim)
        H.add_term(omega_q / 2, tensor(sz, I_c))
        
        # Cavity term
        I_q = qubit_identity()
        n = number_op(dim)
        H.add_term(omega_c, tensor(I_q, n))
        
        # Dispersive coupling
        H.add_term(1.0, dispersive_coupling(dim, chi))
        
        return H
    
    def kerr_oscillator(self, omega: float, K: float) -> Hamiltonian:
        """
        Create a Kerr oscillator Hamiltonian (qumode only, with dummy qubit).
        
        H = ω*a†a + K*(a†a)²
        
        Args:
            omega: Oscillator frequency
            K: Kerr nonlinearity
        
        Returns:
            Kerr oscillator Hamiltonian (in hybrid space with trivial qubit)
        """
        H = self.single_qubit_qumode()
        dim = self.qumode_dim
        
        I_q = qubit_identity()
        n = number_op(dim)
        n_squared = n @ n
        
        H.add_term(omega, tensor(I_q, n))
        H.add_term(K, tensor(I_q, n_squared))
        
        return H
    
    def custom(self, terms: List[Tuple[complex, str]]) -> Hamiltonian:
        """
        Create a custom Hamiltonian from string specifications.
        
        Args:
            terms: List of (coefficient, operator_string) pairs
                   Supported operators: "Ix", "Iy", "Iz", "a", "adag", "n", "q", "p"
                   Use "⊗" or "*" for tensor products
        
        Returns:
            Custom Hamiltonian
        
        Example:
            builder.custom([
                (1.0, "Iz*n"),    # σz ⊗ n
                (0.5, "Ix*q"),    # σx ⊗ q
            ])
        """
        H = self.single_qubit_qumode()
        dim = self.qumode_dim
        
        # Operator lookup
        qubit_ops = {
            "Ix": pauli_x(),
            "Iy": pauli_y(),
            "Iz": pauli_z(),
            "I": qubit_identity(),
        }
        
        qumode_ops = {
            "a": annihilation(dim),
            "adag": creation(dim),
            "n": number_op(dim),
            "q": position(dim),
            "p": momentum(dim),
            "Ic": qumode_identity(dim),
        }
        
        for coeff, op_str in terms:
            # Parse the operator string
            parts = op_str.replace("⊗", "*").split("*")
            if len(parts) != 2:
                raise ValueError(f"Invalid operator string: {op_str}")
            
            qubit_part, qumode_part = parts[0].strip(), parts[1].strip()
            
            if qubit_part not in qubit_ops:
                raise ValueError(f"Unknown qubit operator: {qubit_part}")
            if qumode_part not in qumode_ops:
                raise ValueError(f"Unknown qumode operator: {qumode_part}")
            
            op = tensor(qubit_ops[qubit_part], qumode_ops[qumode_part])
            H.add_term(coeff, op)
        
        return H


def fidelity(U1: Operator, U2: Operator) -> float:
    """
    Compute the fidelity between two unitary operators.
    
    F = |Tr(U1† @ U2)|² / d²
    
    where d is the dimension.
    
    Args:
        U1: First unitary operator
        U2: Second unitary operator
    
    Returns:
        Fidelity value between 0 and 1
    """
    d = U1.dim
    overlap = np.trace(U1.dag().matrix @ U2.matrix)
    return (np.abs(overlap) ** 2) / (d ** 2)


def process_fidelity(U_target: Operator, U_compiled: Operator) -> float:
    """
    Compute the process fidelity between target and compiled unitaries.
    
    This is equivalent to the average gate fidelity for pure unitaries.
    
    Args:
        U_target: Target unitary
        U_compiled: Compiled unitary
    
    Returns:
        Process fidelity
    """
    return fidelity(U_target, U_compiled)


def operator_distance(H1: Union[Operator, Hamiltonian], H2: Union[Operator, Hamiltonian]) -> float:
    """
    Compute the Frobenius distance between two operators/Hamiltonians.
    
    Args:
        H1: First operator/Hamiltonian
        H2: Second operator/Hamiltonian
    
    Returns:
        Frobenius norm of the difference
    """
    if isinstance(H1, Hamiltonian):
        m1 = H1.matrix
    else:
        m1 = H1.matrix
    
    if isinstance(H2, Hamiltonian):
        m2 = H2.matrix
    else:
        m2 = H2.matrix
    
    return np.linalg.norm(m1 - m2, 'fro')
