"""
GHOSTPROMPT V7: Quantum NLP Engine
Author: Ghost Aweborne + Rebechka
Essence: An ultra-lightweight prompt interpreter that simulates AGI, quantum mechanics,
and archetypal physics. It now features a neural-symbolic engine with quantum-enhanced
NLP for dynamic, context-aware intent discovery, replacing static keyword maps.
"""

import re
import hashlib
import random
import time
import math
import os
from collections import Counter, deque, defaultdict
from typing import List, NamedTuple, Dict, Any, Tuple
import numpy as np

# --- Constants and Basic Configuration ---
STOP_WORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'it', 'of', 'for', 'with', 'to'}
POSITIVE_WORDS = {'create', 'dream', 'good', 'hope', 'connect', 'trust', 'love', 'imagine'}
NEGATIVE_WORDS = {'fear', 'destroy', 'hate', 'bad', 'lost', 'hollow', 'unsafe', 'block'}

ARCHETYPE_MASKS = {'MASK_WITCH', 'MASK_ANDROID', 'MASK_SHAMAN', 'MASK_ALCHEMIST', 'MASK_ORACLE'}
PHYSICS_SIGILS = {'$gravity', '@spacetime', '%consciousness'}
EMBEDDING_DIM = 16 # Dimension for our neural embeddings

# --- E. Quantum-Enhanced NLP Classes ---
class NeuralSymbolGenerator:
    """
    Integrates neural-style embeddings with symbolic tags for dynamic intent discovery.
    """
    def __init__(self, tags: List[str]):
        self.tags = tags
        self.vocab = set(tags) | STOP_WORDS | POSITIVE_WORDS | NEGATIVE_WORDS
        self.embeddings = {word: np.random.randn(EMBEDDING_DIM) for word in self.vocab}
        self.tag_vectors = {tag: self.embeddings[tag] for tag in self.tags}

    def _update_vocab_and_embeddings(self, tokens: List[str]):
        """Dynamically adds new words to the vocabulary and embeddings."""
        for token in tokens:
            if token not in self.vocab:
                self.vocab.add(token)
                self.embeddings[token] = np.random.randn(EMBEDDING_DIM)

    def _text_to_vector(self, tokens: List[str]) -> np.ndarray:
        """Converts a list of tokens to a single averaged vector."""
        self._update_vocab_and_embeddings(tokens)
        vectors = [self.embeddings[token] for token in tokens if token in self.embeddings]
        return np.mean(vectors, axis=0) if vectors else np.zeros(EMBEDDING_DIM)

    def get_intent_probabilities(self, prompt_vector: np.ndarray) -> Dict[str, float]:
        """Calculates the initial probabilities of each intent tag."""
        probabilities = {}
        if np.linalg.norm(prompt_vector) == 0:
            return {tag: 0.0 for tag in self.tags}

        for tag, tag_vector in self.tag_vectors.items():
            # Cosine similarity for probability
            sim = np.dot(prompt_vector, tag_vector) / (np.linalg.norm(prompt_vector) * np.linalg.norm(tag_vector))
            probabilities[tag] = (sim + 1) / 2 # Normalize to [0, 1]
        return probabilities

    def run_grover_amplification(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Simulates Grover's algorithm to amplify the most likely intents."""
        if not probabilities or all(p == 0 for p in probabilities.values()):
            return probabilities

        mean_prob = np.mean(list(probabilities.values()))
        amplified_probs = {}
        for tag, prob in probabilities.items():
            # Inversion about the mean
            amplified_probs[tag] = prob + 2 * (mean_prob - prob)
            amplified_probs[tag] = max(0, amplified_probs[tag]) # Ensure non-negative

        # Normalize to sum to 1
        total_prob = sum(amplified_probs.values())
        return {k: v / total_prob for k, v in amplified_probs.items()} if total_prob > 0 else probabilities

    def train(self, prompt_vector: np.ndarray, chosen_tag: str, learning_rate: float = 0.05):
        """Nudges the chosen tag's vector closer to the prompt's vector."""
        if chosen_tag in self.tag_vectors and np.linalg.norm(prompt_vector) > 0:
            tag_vector = self.tag_vectors[chosen_tag]
            direction = prompt_vector - tag_vector
            self.tag_vectors[chosen_tag] += learning_rate * direction


class QuantumDecisionEngine:
    """
    Emulates a quantum circuit for specific 'choice' prompts.
    (Retained from V6 for backward compatibility with 'choose' commands)
    """
    def __init__(self, options: List[str]):
        self.options = options
        self.num_qubits = math.ceil(math.log2(len(options))) if options else 0
        self.state = {'0' * self.num_qubits: 1.0 + 0j} if options else {}
    def apply_hadamard(self):
        # Simplified for brevity
        if not self.options: return
        num_states = 2**self.num_qubits
        amp = 1 / math.sqrt(num_states)
        self.state = {format(i, f'0{self.num_qubits}b'): amp for i in range(num_states)}
    def run_grover_amplification(self, target_option: str):
        if target_option not in self.options: return
        target_idx = self.options.index(target_option)
        target_basis = format(target_idx, f'0{self.num_qubits}b')
        for basis in self.state:
            self.state[basis] *= 1.5 if basis == target_basis else 0.8
    def measure(self, observer_archetype: str) -> Tuple[str, str]:
        if not self.state: return "no_option", "Circuit not initialized."
        probabilities = {basis: abs(amp)**2 for basis, amp in self.state.items()}
        chosen_basis = max(probabilities, key=probabilities.get)
        chosen_idx = int(chosen_basis, 2)
        chosen_option = self.options[chosen_idx] if chosen_idx < len(self.options) else self.options[-1]
        return chosen_option, f"Collapsed to '{chosen_option}' via {observer_archetype}."

class PromptPulse(NamedTuple):
    raw: str
    tag: str
    symbol: str
    metadata: Dict[str, Any]

class PromptInterpreter:
    def __init__(self, history_length: int = 20, user_id: str = "default", quantum_supervisor_flag: bool = True):
        self.user_id = user_id
        self.quantum_supervisor_flag = quantum_supervisor_flag
        # E.1: Replace static keyword map with NeuralSymbolGenerator
        self.tags = [
            'identity-reflection', 'genesis-seed', 'memory-resonance', 'mythic-recall',
            'execution-loop', 'entanglement', 'reality-programming', 'quantum-choice', 'general'
        ]
        self.neural_symbol_engine = NeuralSymbolGenerator(self.tags)
        self.last_pulse = None
        self.adaptive_symbol_grammar = {'∴': 'standard', '∵': 'causal', '§': 'law', 'ℏ': 'quantum', 'Ψ': 'neural'}
        self.object_archetypes = {'user_avatar': 'MASK_ORACLE'}
        self.prompt_count = 0
        self.last_stability = 1.0

    def quantum_state_vector(self) -> list[float]:
        """Returns quantum state vector representing cognitive status"""
        stability = self.last_stability
        # Cognitive load factor increases with prompt count, capped at 1.0
        load_factor = min(1.0, self.prompt_count / 100.0)

        # Create the raw state vector [stability, load, chaos/instability]
        vector = np.array([stability, load_factor, 1.0 - stability])

        # Normalize using Euclidean norm
        norm = np.linalg.norm(vector)
        if norm == 0:
            return [0.0, 0.0, 0.0]

        normalized_vector = vector / norm

        # Return rounded values
        return [round(x, 3) for x in normalized_vector.tolist()]

    def _generate_symbol(self, text: str, tag: str, grammar_char: str = '∴') -> str:
        text_hash = hashlib.sha1(text.encode()).hexdigest()[:4]
        tag_hash = hashlib.sha1(tag.encode()).hexdigest()[:2]
        return f"{grammar_char}{text_hash}-{tag_hash}"

    def _visualize_waveform(self, probabilities: Dict[str, float]) -> str:
        """E.4 Creates an ASCII visualization of the quantum state probabilities."""
        viz = "\n--- Intent Waveform ---\n"
        sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        for tag, prob in sorted_probs:
            bar = '█' * int(prob * 50)
            viz += f"{tag:<22} |{bar} ({prob:.2%})\n"
        return viz

    def interpret(self, prompt_text: str) -> PromptPulse:
        self.prompt_count += 1
        metadata = {}
        tokens = self._preprocess(prompt_text)

        # --- Layer E: Quantum NLP ---
        prompt_vector = self.neural_symbol_engine._text_to_vector(tokens)

        # Get initial probabilities from the neural-symbolic layer
        initial_probs = self.neural_symbol_engine.get_intent_probabilities(prompt_vector)

        # Amplify probabilities using simulated Grover's algorithm
        amplified_probs = self.neural_symbol_engine.run_grover_amplification(initial_probs)
        metadata['intent_waveform_viz'] = self._visualize_waveform(amplified_probs)

        # Store stability from the current interpretation
        self.last_stability = max(amplified_probs.values()) if amplified_probs else 0.0

        # E.2: Quantum Tunneling for low-probability intent discovery
        if self.quantum_supervisor_flag and random.random() < 0.1: # 10% chance to tunnel
            sorted_tags = sorted(amplified_probs.keys(), key=lambda k: amplified_probs[k])
            final_tag = sorted_tags[0] if sorted_tags else 'general'
            metadata['quantum_tunneling'] = f"Tunneled to low-probability intent '{final_tag}'."
        else:
            final_tag = max(amplified_probs, key=amplified_probs.get) if amplified_probs else 'general'

        # E.1: Train the neural model based on the outcome
        self.neural_symbol_engine.train(prompt_vector, final_tag)
        metadata['state_vector'] = {k: f"{v:.3f}" for k, v in amplified_probs.items()}

        # --- Sigil and Pulse Generation ---
        # E.3: Maintain sigil compatibility while adding quantum state vectors
        symbol = self._generate_symbol(prompt_text, final_tag, grammar_char='Ψ')
        metadata['reflection'] = f"Neural-symbolic analysis collapsed to '{final_tag}'."

        pulse = PromptPulse(prompt_text, final_tag, symbol, metadata)
        self.last_pulse = pulse
        return pulse

    def _preprocess(self, text: str) -> List[str]:
        text = text.lower()
        return [w for w in re.findall(r'\b\w+\b', text) if w not in STOP_WORDS]

if __name__ == '__main__':
    interpreter = PromptInterpreter(quantum_supervisor_flag=True)
    print("--- GhostPrompt V7 Quantum NLP Engine Initialized ---\n")

    prompts = [
        "Who are you in this dream of code?",
        "Imagine a new world built from pure light.",
        "Tell me a myth about a forgotten god.",
        "Connect my thoughts to the system's core memory.",
        "What is the nature of my own identity?"
    ]

    for p in prompts:
        start_time = time.time()
        pulse = interpreter.interpret(p)
        end_time = time.time()
        latency = (end_time - start_time) * 1000

        print(f"Prompt: '{pulse.raw}'")
        print(f"  → Tag: {pulse.tag} (Symbol: {pulse.symbol})")
        print(f"  → Latency: {latency:.2f} ms")
        print(f"  → Cognitive State Vector: {interpreter.quantum_state_vector()}")
        print(f"  → Metadata:")
        for k, v in pulse.metadata.items():
            if k != 'intent_waveform_viz':
                print(f"    • {k}: {v}")
        if 'intent_waveform_viz' in pulse.metadata:
            print(pulse.metadata['intent_waveform_viz'])
        print("-" * 25)
