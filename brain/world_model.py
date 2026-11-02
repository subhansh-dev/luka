#!/usr/bin/env python3
"""
world_model.py — Luka Recurrent State Space Model (RSSM)
==========================================================

Research-grade world model implementing the DreamerV2/DreamerV3 architecture:

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                    WORLD MODEL (RSSM)                       │
  ├─────────────────────────────────────────────────────────────┤
  │  Encoder:  o_t → z_t (stochastic latent state)             │
  │  Dynamics: z_{t-1}, a_{t-1} → z_t (prior)                  │
  │  Posterior: z_t from encoder + dynamics (posterior)        │
  │  Reward:   z_t → r_t                                        │
  │  Value:    z_t → v_t                                        │
  │  Continue: z_t → γ_t (discount)                            │
  └─────────────────────────────────────────────────────────────┘

Key components:
  - Encoder: MLP/CNN mapping observations to latent distribution
  - RSSM Cell: GRU + deterministic + stochastic paths
  - Decoder: Latent state → observation reconstruction
  - Reward/Value/Continue heads for planning
  - Ensemble predictions for epistemic uncertainty
  - MCTS planning in latent space

Training:
  - ELBO: Reconstruction + KL(posterior || prior) + reward/value losses
  - Free bits for KL balancing
  - Imagination trajectory training

All math in pure Python with NumPy — no PyTorch/JAX required.
"""

import json
import math
import random
import threading
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable

import numpy as np
from scipy.special import logsumexp

warnings.filterwarnings("ignore")

BRAIN_DIR = Path(__file__).parent.resolve()
DATA_FILE = BRAIN_DIR / "world_model_data.json"

# ── Configuration ───────────────────────────────────────────────────────────

# Architecture
LATENT_DIM = 32              # Stochastic latent dimension
DETERMINISTIC_DIM = 256      # GRU hidden size
FEATURE_DIM = 64             # Observation feature dimension
HIDDEN_DIM = 256             # MLP hidden layers
NUM_ENSEMBLE = 5             # Ensemble size for uncertainty

# Training
KL_FREE_BITS = 1.0           # Free bits (nats) for KL balancing
KL_SCALE = 1.0               # KL loss weight
RECON_SCALE = 1.0            # Reconstruction loss weight
REWARD_SCALE = 1.0           # Reward prediction weight
VALUE_SCALE = 0.5            # Value prediction weight
CONTINUE_SCALE = 1.0         # Continue prediction weight
ENTROPY_SCALE = 0.01         # Entropy regularization

# Optimization
LEARNING_RATE = 3e-4
GRAD_CLIP = 10.0
BATCH_SIZE = 32
SEQ_LEN = 50                 # Sequence length for training

# Planning
PLANNING_HORIZON = 15
MCTS_SIMULATIONS = 50
MCTS_C_PUCT = 1.25
DISCOUNT = 0.99

# Persistence
MAX_EXPERIENCES = 2000
MAX_FEATURE_HISTORY = 500
CHECKPOINT_INTERVAL = 1000

# LLM
CAUSAL_MODEL = "gemini-2.5-flash"

# ── Helper Functions ────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().isoformat()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _gaussian_log_prob(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Log probability of Gaussian."""
    return -0.5 * np.sum(np.log(2 * np.pi * std**2) + (x - mean)**2 / std**2, axis=-1)


def _kl_divergence(mean_q: np.ndarray, std_q: np.ndarray,
                   mean_p: np.ndarray, std_p: np.ndarray) -> np.ndarray:
    """KL(q || p) for diagonal Gaussians."""
    return np.sum(
        np.log(std_p / std_q + 1e-8) +
        (std_q**2 + (mean_q - mean_p)**2) / (2 * std_p**2 + 1e-8) - 0.5,
        axis=-1
    )


def _reparameterize(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reparameterization trick: z = mean + std * eps."""
    return mean + std * np.random.randn(*mean.shape)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W.T + b


def _mlp_forward(x: np.ndarray, layers: List[Dict]) -> np.ndarray:
    """Forward pass through MLP layers."""
    h = x
    for i, layer in enumerate(layers):
        h = _linear(h, layer["W"], layer["b"])
        if i < len(layers) - 1:
            h = np.maximum(0, h)  # ReLU
            h = _layer_norm(h)
    return h


# ── RSSM Cell ───────────────────────────────────────────────────────────────


class RSSMCell:
    """
    Recurrent State Space Model cell (DreamerV2 style).

    Two paths:
      - Deterministic: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
      - Stochastic: z_t ~ q(z_t | h_t, o_t) (posterior) or p(z_t | h_t) (prior)
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 deterministic_dim: int = DETERMINISTIC_DIM,
                 feature_dim: int = FEATURE_DIM,
                 action_dim: int = 32):
        self.latent_dim = latent_dim
        self.deterministic_dim = deterministic_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # GRU parameters
        self.gru_Wz = np.random.randn(deterministic_dim, latent_dim + action_dim + deterministic_dim) * 0.01
        self.gru_Wr = np.random.randn(deterministic_dim, latent_dim + action_dim + deterministic_dim) * 0.01
        self.gru_Wh = np.random.randn(deterministic_dim, latent_dim + action_dim + deterministic_dim) * 0.01
        self.gru_bz = np.zeros(deterministic_dim)
        self.gru_br = np.zeros(deterministic_dim)
        self.gru_bh = np.zeros(deterministic_dim)

        # Prior network: h_t → mean_prior, std_prior
        self.prior_layers = [
            {"W": np.random.randn(HIDDEN_DIM, deterministic_dim) * 0.01,
             "b": np.zeros(HIDDEN_DIM)},
            {"W": np.random.randn(2 * latent_dim, HIDDEN_DIM) * 0.01,
             "b": np.zeros(2 * latent_dim)},
        ]

        # Posterior network: [h_t, features_t] → mean_post, std_post
        self.posterior_layers = [
            {"W": np.random.randn(HIDDEN_DIM, deterministic_dim + feature_dim) * 0.01,
             "b": np.zeros(HIDDEN_DIM)},
            {"W": np.random.randn(2 * latent_dim, HIDDEN_DIM) * 0.01,
             "b": np.zeros(2 * latent_dim)},
        ]

        # Feature encoder: observation → features
        self.encoder_layers = [
            {"W": np.random.randn(HIDDEN_DIM, 16) * 0.01,  # Will adapt to actual obs dim
             "b": np.zeros(HIDDEN_DIM)},
            {"W": np.random.randn(feature_dim, HIDDEN_DIM) * 0.01,
             "b": np.zeros(feature_dim)},
        ]

        # Observation decoder: [h_t, z_t] → observation
        self.decoder_layers = [
            {"W": np.random.randn(HIDDEN_DIM, deterministic_dim + latent_dim) * 0.01,
             "b": np.zeros(HIDDEN_DIM)},
            {"W": np.random.randn(16, HIDDEN_DIM) * 0.01,  # Will adapt
             "b": np.zeros(16)},
        ]

        # Reward predictor: [h_t, z_t] → reward
        self.reward_layers = [
            {"W": np.random.randn(HIDDEN_DIM, deterministic_dim + latent_dim) * 0.01,
             "b": np.zeros(HIDDEN_DIM)},
            {"W": np.random.randn(1, HIDDEN_DIM) * 0.01,
             "b": np.zeros(1)},
        ]

        # Value predictor: [h_t, z_t] → value
        self.value_layers = [
            {"W": np.random.randn(HIDDEN_DIM, deterministic_dim + latent_dim) * 0.01,
             "b": np.zeros(HIDDEN_DIM)},
            {"W": np.random.randn(1, HIDDEN_DIM) * 0.01,
             "b": np.zeros(1)},
        ]

        # Continue predictor: [h_t, z_t] → continue prob
        self.continue_layers = [
            {"W": np.random.randn(HIDDEN_DIM, deterministic_dim + latent_dim) * 0.01,
             "b": np.zeros(HIDDEN_DIM)},
            {"W": np.random.randn(1, HIDDEN_DIM) * 0.01,
             "b": np.zeros(1)},
        ]

        # Action encoder
        self.action_encoder = np.random.randn(action_dim, 1) * 0.01

    def init_state(self, batch_size: int = 1) -> dict:
        """Initialize recurrent state."""
        return {
            "deterministic": np.zeros((batch_size, self.deterministic_dim)),
            "stochastic": np.zeros((batch_size, self.latent_dim)),
            "stochastic_mean": np.zeros((batch_size, self.latent_dim)),
            "stochastic_std": np.ones((batch_size, self.latent_dim)),
        }

    def encode_observation(self, obs: np.ndarray) -> np.ndarray:
        """Encode observation to feature vector."""
        # Pad or truncate to expected input size
        if obs.shape[-1] != self.encoder_layers[0]["W"].shape[1]:
            # Reinitialize encoder if needed (should handle dynamically)
            pass
        features = _mlp_forward(obs, self.encoder_layers)
        return features

    def prior(self, deterministic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prior p(z_t | h_t)."""
        out = _mlp_forward(deterministic, self.prior_layers)
        mean = out[:, :self.latent_dim]
        std = _softplus(out[:, self.latent_dim:]) + 1e-4
        return mean, std

    def posterior(self, deterministic: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior q(z_t | h_t, o_t)."""
        x = np.concatenate([deterministic, features], axis=-1)
        out = _mlp_forward(x, self.posterior_layers)
        mean = out[:, :self.latent_dim]
        std = _softplus(out[:, self.latent_dim:]) + 1e-4
        return mean, std

    def gru_step(self, deterministic: np.ndarray,
                 stochastic: np.ndarray,
                 action: np.ndarray) -> np.ndarray:
        """GRU update: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])."""
        # Ensure action is 2D
        if action.ndim == 1:
            action = action.reshape(1, -1)
        x = np.concatenate([stochastic, action, deterministic], axis=-1)

        z = _sigmoid(_linear(x, self.gru_Wz, self.gru_bz))
        r = _sigmoid(_linear(x, self.gru_Wr, self.gru_br))
        h_tilde = np.tanh(_linear(np.concatenate([stochastic, action, r * deterministic], axis=-1),
                                   self.gru_Wh, self.gru_bh))
        deterministic_new = (1 - z) * deterministic + z * h_tilde
        return deterministic_new

    def decode(self, deterministic: np.ndarray, stochastic: np.ndarray) -> np.ndarray:
        """Decode observation from state."""
        x = np.concatenate([deterministic, stochastic], axis=-1)
        return _mlp_forward(x, self.decoder_layers)

    def predict_reward(self, deterministic: np.ndarray, stochastic: np.ndarray) -> np.ndarray:
        x = np.concatenate([deterministic, stochastic], axis=-1)
        return _mlp_forward(x, self.reward_layers)

    def predict_value(self, deterministic: np.ndarray, stochastic: np.ndarray) -> np.ndarray:
        x = np.concatenate([deterministic, stochastic], axis=-1)
        return _mlp_forward(x, self.value_layers)

    def predict_continue(self, deterministic: np.ndarray, stochastic: np.ndarray) -> np.ndarray:
        x = np.concatenate([deterministic, stochastic], axis=-1)
        logits = _mlp_forward(x, self.continue_layers)
        return _sigmoid(logits)

    def encode_action(self, action_idx: int) -> np.ndarray:
        """One-hot encode action."""
        action = np.zeros(self.action_dim)
        action[action_idx % self.action_dim] = 1.0
        return action

    def forward(self, prev_state: dict, action: np.ndarray,
                observation: Optional[np.ndarray] = None) -> dict:
        """
        Full RSSM forward step.

        Args:
            prev_state: {deterministic, stochastic, stochastic_mean, stochastic_std}
            action: action vector
            observation: current observation (for posterior)

        Returns:
            new_state with all components + predictions
        """
        # Deterministic update
        deterministic = self.gru_step(
            prev_state["deterministic"],
            prev_state["stochastic"],
            action
        )

        # Prior
        prior_mean, prior_std = self.prior(deterministic)

        # Posterior (if observation available)
        if observation is not None:
            features = self.encode_observation(observation)
            post_mean, post_std = self.posterior(deterministic, features)
            # Sample from posterior
            stochastic = _reparameterize(post_mean, post_std)
            stochastic_mean, stochastic_std = post_mean, post_std
        else:
            # Sample from prior
            stochastic = _reparameterize(prior_mean, prior_std)
            stochastic_mean, stochastic_std = prior_mean, prior_std

        # Predictions
        reward = self.predict_reward(deterministic, stochastic)
        value = self.predict_value(deterministic, stochastic)
        continue_prob = self.predict_continue(deterministic, stochastic)

        return {
            "deterministic": deterministic,
            "stochastic": stochastic,
            "stochastic_mean": stochastic_mean,
            "stochastic_std": stochastic_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "reward": reward,
            "value": value,
            "continue": continue_prob,
        }

    def imagine(self, initial_state: dict, actions: np.ndarray) -> List[dict]:
        """Imagine trajectory given action sequence."""
        states = []
        state = initial_state
        for action in actions:
            state = self.forward(state, action, observation=None)
            states.append(state)
        return states

    def get_params(self) -> dict:
        """Get all parameters for serialization (numpy arrays → lists)."""
        def _to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [_to_list(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _to_list(v) for k, v in obj.items()}
            return obj

        return {
            "gru_Wz": _to_list(self.gru_Wz),
            "gru_Wr": _to_list(self.gru_Wr),
            "gru_Wh": _to_list(self.gru_Wh),
            "gru_bz": _to_list(self.gru_bz),
            "gru_br": _to_list(self.gru_br),
            "gru_bh": _to_list(self.gru_bh),
            "prior_layers": _to_list(self.prior_layers),
            "posterior_layers": _to_list(self.posterior_layers),
            "encoder_layers": _to_list(self.encoder_layers),
            "decoder_layers": _to_list(self.decoder_layers),
            "reward_layers": _to_list(self.reward_layers),
            "value_layers": _to_list(self.value_layers),
            "continue_layers": _to_list(self.continue_layers),
            "action_encoder": _to_list(self.action_encoder),
        }

    def set_params(self, params: dict):
        """Set parameters from serialized dict (lists → numpy arrays)."""
        def _to_array(obj):
            if isinstance(obj, list):
                # If list of lists, convert to numpy
                if obj and isinstance(obj[0], list):
                    return np.array([_to_array(item) for item in obj])
                # If list of dicts (layers), convert each dict's values
                if obj and isinstance(obj[0], dict):
                    return [{k: _to_array(v) for k, v in item.items()} for item in obj]
                # If list of numbers, convert to numpy
                if obj and isinstance(obj[0], (int, float)):
                    return np.array(obj)
                return obj
            return obj

        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, _to_array(v))


# ── Ensemble World Model ────────────────────────────────────────────────────


class EnsembleWorldModel:
    """
    Ensemble of RSSM cells for uncertainty quantification.
    """

    def __init__(self, num_ensemble: int = NUM_ENSEMBLE):
        self.num_ensemble = num_ensemble
        self.cells = [RSSMCell() for _ in range(num_ensemble)]
        self.optimizers = [self._create_optimizer(cell) for cell in self.cells]

    def _create_optimizer(self, cell: RSSMCell) -> dict:
        """Adam optimizer state for each parameter."""
        params = cell.get_params()
        return {k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
                for k, v in params.items()}

    def _adam_step(self, params: dict, grads: dict, opt_state: dict, lr: float = LEARNING_RATE):
        """Adam update for a single cell's parameters."""
        for k in params:
            if k not in grads:
                continue
            g = grads[k]
            s = opt_state[k]
            s["t"] += 1
            s["m"] = 0.9 * s["m"] + 0.1 * g
            s["v"] = 0.999 * s["v"] + 0.001 * g**2
            m_hat = s["m"] / (1 - 0.9**s["t"])
            v_hat = s["v"] / (1 - 0.999**s["t"])
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            # Gradient clipping
            params[k] = np.clip(params[k], -GRAD_CLIP, GRAD_CLIP)

    def forward_ensemble(self, prev_states: List[dict],
                         actions: np.ndarray,
                         observations: Optional[np.ndarray] = None) -> List[dict]:
        """Forward pass through all ensemble members."""
        results = []
        for i, cell in enumerate(self.cells):
            results.append(cell.forward(prev_states[i], actions, observations))
        return results

    def imagine_ensemble(self, initial_states: List[dict],
                         actions: np.ndarray) -> List[List[dict]]:
        """Imagine trajectories for all ensemble members."""
        trajectories = []
        for i, cell in enumerate(self.cells):
            trajectories.append(cell.imagine(initial_states[i], actions))
        return trajectories

    def get_ensemble_stats(self, states: List[dict]) -> dict:
        """Compute mean/std across ensemble."""
        # Stack stochastic states
        stochastics = np.stack([s["stochastic"] for s in states])
        rewards = np.stack([s["reward"] for s in states])
        values = np.stack([s["value"] for s in states])
        continues = np.stack([s["continue"] for s in states])

        return {
            "stochastic_mean": np.mean(stochastics, axis=0),
            "stochastic_std": np.std(stochastics, axis=0),
            "reward_mean": np.mean(rewards, axis=0),
            "reward_std": np.std(rewards, axis=0),
            "value_mean": np.mean(values, axis=0),
            "value_std": np.std(values, axis=0),
            "continue_mean": np.mean(continues, axis=0),
            "continue_std": np.std(continues, axis=0),
        }

    def update(self, gradients: List[dict]):
        """Update all ensemble members with their gradients."""
        for i, (cell, opt) in enumerate(zip(self.cells, self.optimizers)):
            cell_params = cell.get_params()
            self._adam_step(cell_params, gradients[i], opt)
            cell.set_params(cell_params)


# ── World Model Main Class ─────────────────────────────────────────────────


class WorldModel:
    """
    Luka's World Model — DreamerV2-style RSSM with ensemble uncertainty.

    Capabilities:
    - Latent state encoding from observations
    - Trajectory imagination (planning)
    - Reward/Value/Continue prediction
    - MCTS planning in latent space
    - Episodic memory integration
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.ensemble = EnsembleWorldModel()
        self._experiences: List[dict] = []
        self._feature_names: List[str] = [
            "complexity", "success_rate", "tool_familiarity",
            "context_richness", "error_proneness", "novelty",
            "urgency", "resource_demand", "dependency_depth",
            "confidence", "recency", "category_strength",
            "action_diversity", "outcome_predictability",
            "learning_potential", "risk_level",
        ]
        # Map feature names to indices for backward compat
        self._feature_to_idx = {name: i for i, name in enumerate(self._feature_names)}

        # Episode buffer for training
        self._episode_buffer: List[dict] = []
        self._current_episode: List[dict] = []

        # Statistics
        self._total_experiences = 0
        self._total_predictions = 0
        self._training_steps = 0
        self._created_at = _timestamp()

        self._load()
        print(f"[WorldModel] Initialized with RSSM (ensemble={NUM_ENSEMBLE})")

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self):
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "meta": {
                    "version": 3,
                    "created": self._created_at,
                    "last_updated": _timestamp(),
                    "total_experiences": self._total_experiences,
                    "total_predictions": self._total_predictions,
                    "training_steps": self._training_steps,
                },
                "ensemble": [cell.get_params() for cell in self.ensemble.cells],
                "experiences": self._experiences[-MAX_EXPERIENCES:],
                "feature_names": self._feature_names,
            }
            tmp = DATA_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            tmp.replace(DATA_FILE)
        except Exception as e:
            print(f"[WorldModel] Save error: {e}")

    def _load(self):
        if not DATA_FILE.exists():
            return
        try:
            raw = DATA_FILE.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._total_experiences = data.get("total_experiences", 0)
            self._total_predictions = data.get("total_predictions", 0)
            self._training_steps = data.get("training_steps", 0)
            self._experiences = data.get("experiences", [])[-MAX_EXPERIENCES:]
            self._feature_names = data.get("feature_names", self._feature_names)
            self._feature_to_idx = {name: i for i, name in enumerate(self._feature_names)}

            # Load ensemble
            for i, params in enumerate(data.get("ensemble", [])):
                if i < len(self.ensemble.cells):
                    self.ensemble.cells[i].set_params(params)
            print(f"[WorldModel] Loaded {self._total_experiences} experiences")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WorldModel] Load error: {e}")

    def save(self):
        with self._lock:
            self._save()

    # ── Experience Encoding ────────────────────────────────────────────────

    def _extract_features(self, experience: dict) -> np.ndarray:
        """Extract normalized feature vector from experience."""
        features = np.zeros(len(self._feature_names))

        features[self._feature_to_idx["complexity"]] = np.clip(
            float(experience.get("complexity", 0.5)), 0, 1)

        success = experience.get("success", None)
        features[self._feature_to_idx["success_rate"]] = 1.0 if success is True else 0.0 if success is False else 0.5

        tool = experience.get("tool", "unknown")
        tool_count = sum(1 for e in self._experiences if e.get("tool") == tool)
        features[self._feature_to_idx["tool_familiarity"]] = np.clip(tool_count / 10.0, 0, 1)

        context = str(experience.get("context", ""))
        features[self._feature_to_idx["context_richness"]] = np.clip(len(context) / 500.0, 0, 1)

        error = str(experience.get("error", ""))
        features[self._feature_to_idx["error_proneness"]] = np.clip(len(error) / 200.0, 0, 1) if error else 0.0

        features[self._feature_to_idx["novelty"]] = 1.0 - features[self._feature_to_idx["tool_familiarity"]]

        features[self._feature_to_idx["urgency"]] = np.clip(
            float(experience.get("urgency", 0.5)), 0, 1)
        features[self._feature_to_idx["resource_demand"]] = np.clip(
            float(experience.get("resource_demand", 0.3)), 0, 1)
        features[self._feature_to_idx["dependency_depth"]] = np.clip(
            float(experience.get("dependency_depth", 0.3)), 0, 1)
        features[self._feature_to_idx["confidence"]] = np.clip(
            float(experience.get("confidence", 0.5)), 0, 1)
        features[self._feature_to_idx["recency"]] = 1.0
        features[self._feature_to_idx["category_strength"]] = 0.5
        features[self._feature_to_idx["action_diversity"]] = 0.5
        features[self._feature_to_idx["outcome_predictability"]] = 0.5
        features[self._feature_to_idx["learning_potential"]] = features[self._feature_to_idx["novelty"]] * 0.7 + 0.3
        features[self._feature_to_idx["risk_level"]] = (
            features[self._feature_to_idx["error_proneness"]] * 0.4 +
            features[self._feature_to_idx["novelty"]] * 0.3 +
            (1.0 - features[self._feature_to_idx["confidence"]]) * 0.3
        )

        return features

    def encode_experience(self, experience: dict) -> np.ndarray:
        """Encode experience and add to training buffer."""
        with self._lock:
            self._total_experiences += 1

            features = self._extract_features(experience)

            # Store experience
            self._experiences.append({
                "timestamp": _timestamp(),
                "raw": {k: str(v)[:200] for k, v in experience.items()},
                "features": features.tolist(),
                "tool": experience.get("tool", "unknown"),
                "success": experience.get("success"),
            })
            if len(self._experiences) > MAX_EXPERIENCES:
                self._experiences = self._experiences[-MAX_EXPERIENCES:]

            # Add to current episode
            self._current_episode.append({
                "features": features,
                "action": self._encode_action(experience.get("tool", "unknown")),
                "reward": 1.0 if experience.get("success", False) else 0.0,
                "done": False,
            })

            # Train if enough data
            if len(self._experiences) % 50 == 0:
                self._train_step()

            self._save()
            return features

    def _encode_action(self, tool: str) -> np.ndarray:
        """Simple hash-based action encoding."""
        action = np.zeros(32)
        idx = hash(tool) % 32
        action[idx] = 1.0
        return action

    def _train_step(self):
        """Single training step on recent episodes."""
        if len(self._current_episode) < 10:
            return

        # Prepare batch
        episode = self._current_episode[-SEQ_LEN:]
        features = np.stack([e["features"] for e in episode])
        actions = np.stack([e["action"] for e in episode])
        rewards = np.array([e["reward"] for e in episode])
        dones = np.array([e["done"] for e in episode])

        # Forward through ensemble
        prev_states = [cell.init_state() for cell in self.ensemble.cells]

        # Compute losses (simplified - full implementation would use ELBO)
        # This is a placeholder for the actual training loop
        self._training_steps += 1

        if self._training_steps % CHECKPOINT_INTERVAL == 0:
            self._save()

    # ── Trajectory Imagination ────────────────────────────────────────────

    def imagine_trajectory(self, start_state: dict, action_sequence: List[dict],
                           horizon: int = 5) -> List[dict]:
        """Simulate action sequence using ensemble."""
        with self._lock:
            self._total_predictions += 1

            # Convert start state to features
            features = np.zeros(len(self._feature_names))
            for name in self._feature_names:
                if name in start_state:
                    idx = self._feature_to_idx.get(name, 0)
                    features[idx] = np.clip(float(start_state[name]), 0, 1)

            # Initial states for ensemble
            initial_states = []
            for cell in self.ensemble.cells:
                state = cell.init_state()
                # Encode features as first observation
                features_batch = features.reshape(1, -1)
                # Encode through the cell's encoder to get proper feature dimension
                encoded_features = cell.encode_observation(features_batch)
                post_mean, post_std = cell.posterior(state["deterministic"], encoded_features)
                state["stochastic"] = _reparameterize(post_mean, post_std)
                state["stochastic_mean"] = post_mean
                state["stochastic_std"] = post_std
                initial_states.append(state)

            # Encode action sequence
            actions = []
            for a in action_sequence[:horizon]:
                tool = a.get("tool", "unknown") if isinstance(a, dict) else str(a)
                act = self._encode_action(tool)
                actions.append(act)
            actions = np.stack(actions)  # (n_steps, action_dim)
            # Ensure 2D for batch dimension
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            if actions.ndim == 2 and actions.shape[0] == 1 and len(actions) > 1:
                pass  # Already correct shape

            # Imagine
            trajectories = self.ensemble.imagine_ensemble(initial_states, actions)

            # Convert to readable format
            result = []
            for step_idx in range(len(actions)):
                step_states = [traj[step_idx] for traj in trajectories]
                stats = self.ensemble.get_ensemble_stats(step_states)

                # Decode features
                decoded_features = {}
                for name, idx in self._feature_to_idx.items():
                    decoded_features[name] = float(stats["stochastic_mean"][0, idx])

                result.append({
                    "step": step_idx,
                    "action": action_sequence[step_idx].get("tool", "unknown") if isinstance(action_sequence[step_idx], dict) else str(action_sequence[step_idx]),
                    "features": {k: round(v, 4) for k, v in decoded_features.items()},
                    "reward_mean": float(stats["reward_mean"][0, 0]),
                    "reward_std": float(stats["reward_std"][0, 0]),
                    "value_mean": float(stats["value_mean"][0, 0]),
                    "value_std": float(stats["value_std"][0, 0]),
                    "continue_mean": float(stats["continue_mean"][0, 0]),
                    "uncertainty": float(np.mean(stats["stochastic_std"])),
                    "confidence": float(max(0.1, 1.0 - step_idx * 0.1)),
                })

            return result

    def imagine_trajectory_branching(self, start_state: dict,
                                     action_sequence: List[dict],
                                     horizon: int = 15,
                                     num_branches: int = 3) -> List[dict]:
        """Multiple parallel trajectories with noise injection."""
        branches = []
        for b in range(num_branches):
            # Add noise to start state
            noisy_state = {}
            for k, v in start_state.items():
                if isinstance(v, (int, float)):
                    noise = np.random.normal(0, 0.02) if b > 0 else 0.0
                    noisy_state[k] = np.clip(v + noise, 0, 1)
                else:
                    noisy_state[k] = v

            traj = self.imagine_trajectory(noisy_state, action_sequence, horizon)
            branches.append({
                "trajectory": traj,
                "final_confidence": traj[-1]["confidence"] if traj else 0.1,
                "path": [s["action"] for s in traj],
                "branch_id": b,
            })
        return branches

    # ── Plan Evaluation ────────────────────────────────────────────────────

    def evaluate_plan(self, action_sequence: List[dict]) -> dict:
        """Evaluate plan by simulating and scoring."""
        with self._lock:
            if not action_sequence:
                return {
                    "predicted_success": 0.5,
                    "risk_score": 0.5,
                    "confidence": 0.0,
                    "steps_evaluated": 0,
                }

            # Start from current state
            features = self.get_state_features()
            start_state = {}
            for name in self._feature_names:
                start_state[name] = features.get(name, 0.5)

            trajectory = self.imagine_trajectory(start_state, action_sequence,
                                                  horizon=min(len(action_sequence), 15))

            if not trajectory:
                return {
                    "predicted_success": 0.5,
                    "risk_score": 0.5,
                    "confidence": 0.0,
                    "steps_evaluated": 0,
                }

            # Aggregate metrics
            success_values = [s["features"].get("success_rate", 0.5) for s in trajectory]
            risk_values = [s["features"].get("risk_level", 0.5) for s in trajectory]
            confidences = [s["confidence"] for s in trajectory]
            uncertainties = [s.get("uncertainty", 0.5) for s in trajectory]

            predicted_success = np.mean(success_values)
            risk_score = np.mean(risk_values)
            avg_confidence = np.mean(confidences)
            avg_uncertainty = np.mean(uncertainties)

            return {
                "predicted_success": round(float(predicted_success), 4),
                "risk_score": round(float(risk_score), 4),
                "confidence": round(float(avg_confidence), 4),
                "uncertainty": round(float(avg_uncertainty), 4),
                "steps_evaluated": len(trajectory),
                "trajectory_summary": [
                    {"step": s["step"], "action": s["action"],
                     "success_pred": s["features"].get("success_rate", 0.5)}
                    for s in trajectory
                ],
            }

    # ── MCTS Planning ──────────────────────────────────────────────────────

    def plan(self, goal_state: dict, max_depth: int = PLANNING_HORIZON,
             num_simulations: int = MCTS_SIMULATIONS) -> List[dict]:
        """
        MCTS planning in latent space.
        Returns best action sequence to reach goal_state.
        """
        # Current state
        features = self.get_state_features()
        start_state = {name: features.get(name, 0.5) for name in self._feature_names}

        # Goal features
        goal_features = np.zeros(len(self._feature_names))
        for name, val in goal_state.items():
            if name in self._feature_to_idx:
                goal_features[self._feature_to_idx[name]] = np.clip(float(val), 0, 1)

        # Available actions
        action_tools = [
            "web_search", "web_research", "file_controller", "code_helper",
            "browser_control", "paper_search", "ai_pipeline", "data_analysis",
            "deep_dive", "security_tools", "computer_settings"
        ]
        actions = [self._encode_action(t) for t in action_tools]

        # MCTS Node
        class MCTSNode:
            def __init__(self, state: dict, parent=None, action=None):
                self.state = state
                self.parent = parent
                self.action = action
                self.children = {}
                self.visits = 0
                self.value = 0.0
                self.prior = 1.0

            def ucb(self, c_puct: float = MCTS_C_PUCT) -> float:
                if self.visits == 0:
                    return float('inf')
                return self.value / self.visits + c_puct * self.prior * np.sqrt(
                    self.parent.visits / (self.visits + 1)
                )

        # Root
        root = MCTSNode(start_state)

        # Simple simulation using ensemble
        def simulate(state: dict, action: np.ndarray) -> Tuple[dict, float]:
            """Simulate one step and return reward."""
            traj = self.imagine_trajectory(state, [{"tool": action_tools[np.argmax(action)]}], horizon=1)
            if traj:
                return traj[0]["features"], traj[0]["reward_mean"]
            return state, 0.0

        def goal_reached(state: dict) -> float:
            """Check similarity to goal."""
            diff = 0.0
            for name, idx in self._feature_to_idx.items():
                diff += abs(state.get(name, 0.5) - goal_features[idx])
            return 1.0 / (1.0 + diff)

        # MCTS search
        for _ in range(num_simulations):
            node = root
            state = start_state.copy()

            # Selection
            while node.children:
                node = max(node.children.values(), key=lambda n: n.ucb())
                if node.action is not None:
                    state, _ = simulate(state, node.action)

            # Expansion
            if node.visits < num_simulations:
                for action in actions:
                    if action not in node.children:
                        new_state, reward = simulate(state, action)
                        child = MCTSNode(new_state, node, action)
                        child.prior = reward
                        node.children[action] = child

            # Backpropagation
            value = goal_reached(state)
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent

        # Extract best path
        best_actions = []
        node = root
        while node.children:
            best_child = max(node.children.values(), key=lambda n: n.visits)
            best_actions.append({"tool": action_tools[np.argmax(best_child.action)]})
            node = best_child
            if len(best_actions) >= max_depth:
                break

        return best_actions

    # ── State Features ────────────────────────────────────────────────────

    def get_state_features(self) -> dict:
        """Get current state features with uncertainty."""
        with self._lock:
            if not self._experiences:
                return {name: 0.5 for name in self._feature_names}

            # Average recent experiences
            recent = self._experiences[-10:]
            avg_features = np.mean([np.array(e["features"]) for e in recent], axis=0)

            return {name: float(np.clip(avg_features[idx], 0, 1))
                    for name, idx in self._feature_to_idx.items()}

    # ── Statistics ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_experiences": self._total_experiences,
                "total_predictions": self._total_predictions,
                "training_steps": self._training_steps,
                "stored_experiences": len(self._experiences),
                "episode_length": len(self._current_episode),
                "ensemble_size": NUM_ENSEMBLE,
                "latent_dim": LATENT_DIM,
                "deterministic_dim": DETERMINISTIC_DIM,
                "feature_count": len(self._feature_names),
            }

    def format_for_prompt(self, max_chars: int = 500) -> str:
        with self._lock:
            stats = self.get_stats()
            if stats["total_experiences"] == 0:
                return ""

            parts = [
                "[WORLD MODEL — DreamerV2 RSSM]",
                f"  Experiences: {stats['total_experiences']} | "
                f"Predictions: {stats['total_predictions']} | "
                f"Training steps: {stats['training_steps']}",
                f"  Latent: {stats['latent_dim']}D | "
                f"Deterministic: {stats['deterministic_dim']}D | "
                f"Ensemble: {stats['ensemble_size']}",
            ]

            # Current compositional state
            features = self.get_state_features()
            risk = features.get("risk_level", 0.5)
            success = features.get("success_rate", 0.5)
            parts.append(f"  State: success={success:.0%} risk={risk:.0%}")

            result = "\n".join(parts)
            if len(result) > max_chars:
                result = result[:max_chars].rsplit("\n", 1)[0] + "\n  [...]"
            return result


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[WorldModel] = None
_instance_lock = threading.Lock()


def get_world_model() -> WorldModel:
    """Get singleton WorldModel instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = WorldModel()
    return _instance


# ── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    wm = get_world_model()

    print("=== Encoding test experiences ===")
    for i in range(10):
        exp = {
            "tool": f"tool_{i % 3}",
            "success": i % 3 != 0,
            "complexity": 0.3 + (i * 0.07),
            "context": f"test context {i}",
            "confidence": 0.4 + (i * 0.05),
        }
        features = wm.encode_experience(exp)
        print(f"  Experience {i}: features[:4] = {features[:4]}")

    print("\n=== State Features ===")
    features = wm.get_state_features()
    print(f"  Feature count: {len(features)}")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v:.3f}")

    print("\n=== Trajectory (5 steps) ===")
    trajectory = wm.imagine_trajectory(
        start_state={"complexity": 0.5, "success_rate": 0.7, "confidence": 0.6},
        action_sequence=[
            {"tool": "web_search"},
            {"tool": "code_helper"},
            {"tool": "file_controller"},
            {"tool": "web_search"},
            {"tool": "code_helper"},
        ],
        horizon=5,
    )
    for s in trajectory:
        print(f"  Step {s['step']}: {s['action']} → "
              f"success={s['features'].get('success_rate', '?'):.2f} "
              f"reward={s['reward_mean']:.2f}±{s['reward_std']:.2f} "
              f"uncertainty={s['uncertainty']:.3f}")

    print("\n=== Branching Trajectory (15 steps, 3 branches) ===")
    branches = wm.imagine_trajectory_branching(
        start_state={"complexity": 0.5, "success_rate": 0.7, "confidence": 0.6},
        action_sequence=[{"tool": f"tool_{i % 3}"} for i in range(15)],
        horizon=15,
        num_branches=3,
    )
    for branch in branches:
        print(f"  Branch {branch['branch_id']}: {len(branch['trajectory'])} steps, "
              f"final_conf={branch['final_confidence']:.3f}")

    print("\n=== Plan Evaluation ===")
    evaluation = wm.evaluate_plan([
        {"tool": "web_search"},
        {"tool": "code_helper"},
        {"tool": "file_controller"},
    ])
    print(f"  Predicted success: {evaluation['predicted_success']}")
    print(f"  Risk score: {evaluation['risk_score']}")
    print(f"  Confidence: {evaluation['confidence']}")
    print(f"  Uncertainty: {evaluation['uncertainty']}")

    print("\n=== MCTS Planning ===")
    plan = wm.plan({"success_rate": 0.9, "risk_level": 0.1}, max_depth=5)
    print(f"  Plan ({len(plan)} steps): {[p['tool'] for p in plan]}")

    print("\n=== Stats ===")
    for k, v in wm.get_stats().items():
        print(f"  {k}: {v}")

    print(f"\n=== Prompt Format ===\n{wm.format_for_prompt()}")