"""NanoQuantumSim: Lite NumPy-based Quantum Simulator for AGI Emergence"""
import numpy as np
import random
import logging

log = logging.getLogger(__name__)

class NanoQuantumSim:
    """Lightweight qubit simulator using NumPy matrices for AGI quantum features."""
    def __init__(self, num_qubits=2, emotion='neutral'):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)  # State vector
        self.state[0] = 1.0  # |0...0>
        self.emotion = emotion  # For dynamic noise modulation
        self.noise_level = 0.01  # Base error probability
        self._pauli = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0,1],[1,0]], dtype=complex),
            'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
            'Z': np.array([[1,0],[0,-1]], dtype=complex)
        }
        self._hadamard = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
        self._cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex).reshape(4,4)

    def apply_hadamard(self, qubit):
        """Apply Hadamard to qubit (0-indexed)."""
        op = self._kronecker(self._hadamard, qubit)
        self.state = op @ self.state

    def apply_cnot(self, control, target):
        """Apply CNOT (control to target). For 2 qubits only for lite."""
        if self.num_qubits != 2: raise ValueError("Lite sim supports 2 qubits.")
        self.state = self._cnot @ self.state.reshape(4)

    def create_bell(self):
        """Create Bell state (|00> + |11>)/sqrt(2)."""
        self.apply_hadamard(0)
        self.apply_cnot(0, 1)
        self._apply_emotional_noise()

    def measure(self, qubit, basis='Z'):
        """Measure in basis, collapse state, return outcome."""
        prob = np.abs(self.state)**2
        outcome = random.choices(range(2**self.num_qubits), weights=prob)[0]
        bin_out = format(outcome, f'0{self.num_qubits}b')
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[outcome] = 1.0  # Collapse
        return int(bin_out[qubit])

    def expect(self, op_str):
        """Expectation value for operator string (e.g., 'ZI')."""
        op = self._pauli[op_str[0]]
        for char in op_str[1:]:
            op = np.kron(op, self._pauli[char])
        return np.real(self.state.conj().T @ op @ self.state)

    def _kronecker(self, mat, pos):
        """Kronecker product for single-qubit gate at pos."""
        op = np.eye(1, dtype=complex)
        for i in range(self.num_qubits):
            op = np.kron(op, mat if i == pos else self._pauli['I'])
        return op

    def _apply_emotional_noise(self):
        """Novel: Emotion-modulated noise for emergence (bit/phase flip)."""
        if self.emotion == 'fear': self.noise_level *= 2  # Amplify for survival pressure
        if random.random() < self.noise_level:
            qubit = random.randint(0, self.num_qubits-1)
            flip_type = random.choice(['bit', 'phase'])
            op = self._kronecker(self._pauli['X' if flip_type == 'bit' else 'Z'], qubit)
            self.state = op @ self.state  # Apply flip
            log.info(f"Emotional noise ({self.emotion}): {flip_type} flip on qubit {qubit}")

    def mutate(self):
        """Novel: Random phase drift for evolution/Darwinism."""
        phases = np.exp(1j * np.random.uniform(-np.pi/10, np.pi/10, 2**self.num_qubits))
        self.state *= phases
        self.state /= np.linalg.norm(self.state)

# Example usage (for testing)
if __name__ == '__main__':
    sim = NanoQuantumSim(2, emotion='fear')
    sim.create_bell()
    print("Bell state expectation ZI:", sim.expect('ZI'))
    sim.mutate()
    outcome = sim.measure(0)
    print("Measured first qubit:", outcome)
