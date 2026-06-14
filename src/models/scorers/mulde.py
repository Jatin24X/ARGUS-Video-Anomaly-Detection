"""ARGUS - MULDE anomaly scorer (Stream A).

Paper-faithful Phase 1 reproduction of MULDE:
  Micorek et al., "Multiscale Log-Density Estimation via Denoising Score
  Matching for Video Anomaly Detection", CVPR 2024.

The implementation follows the official training recipe in the upstream
`main.py` and `models.py`:
  - feature standardization with training-set mean/std
  - per-sample log-uniform sigma sampling during training
  - denoising score matching target `noise / sigma^2`
  - lambda weighting `lambda(sigma) = sigma^2`
  - optional beta regularization on clean data
  - inference over `L` linearly spaced sigma values
"""

import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)


class MULDENetwork(nn.Module):
    """Log-density network f_theta(x, sigma) -> scalar."""

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 4096,
        num_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = feature_dim + 1
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MULDEScorer(nn.Module):
    """Multi-scale log-density estimator for one-class anomaly scoring."""

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 4096,
        sigma_low: float = 1e-3,
        sigma_high: float = 1.0,
        eval_L: int = 16,
        beta: float = 0.0,
        gmm_components: int = 5,
        num_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.eval_L = eval_L
        self.beta = beta or 0.0
        self.gmm_components = gmm_components
        self.use_layernorm = use_layernorm

        self.network = MULDENetwork(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_layernorm=use_layernorm,
        )

        self.register_buffer("feat_mean", torch.zeros(feature_dim))
        self.register_buffer("feat_std", torch.ones(feature_dim))

        self._gmm = None
        self._train_log_densities = None
        self._legacy_eval_sigmas: Optional[List[float]] = None

        logger.info(
            "MULDEScorer: feature_dim=%s hidden_dim=%s sigma=[%s, %s] eval_L=%s "
            "beta=%s gmm_components=%s layernorm=%s",
            feature_dim,
            hidden_dim,
            sigma_low,
            sigma_high,
            eval_L,
            self.beta,
            gmm_components,
            use_layernorm,
        )

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set feature standardization stats computed from training features."""
        if mean.shape != self.feat_mean.shape or std.shape != self.feat_std.shape:
            raise ValueError(
                f"Feature stats shape mismatch: mean={tuple(mean.shape)} "
                f"std={tuple(std.shape)} expected={tuple(self.feat_mean.shape)}"
            )
        self.feat_mean.copy_(mean.to(self.feat_mean.device, dtype=self.feat_mean.dtype))
        self.feat_std.copy_(std.to(self.feat_std.device, dtype=self.feat_std.dtype))

    def _standardize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize features using training-set statistics."""
        return (features - self.feat_mean) / (self.feat_std + 1e-8)

    def _get_eval_sigmas(self) -> np.ndarray:
        """Return inference sigmas, preserving legacy discrete checkpoints."""
        if self._legacy_eval_sigmas:
            return np.asarray(self._legacy_eval_sigmas, dtype=np.float32)
        return np.linspace(self.sigma_low, self.sigma_high, self.eval_L, dtype=np.float32)

    def _score_with_log_density(
        self,
        net_input: torch.Tensor,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute score = grad(-log_density) and return both score and density."""
        net_input = net_input.requires_grad_(True)
        log_density = self.network(net_input)
        logp = -log_density.sum()
        score = torch.autograd.grad(logp, net_input, create_graph=create_graph)[0]
        return score, log_density

    def compute_score_and_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Denoising score matching loss matching the official MULDE code."""
        batch_size = features.shape[0]
        device = features.device

        x = self._standardize_features(features)
        x = x.requires_grad_(True)

        log_sigma = torch.empty(batch_size, 1, device=device).uniform_(
            math.log(self.sigma_low),
            math.log(self.sigma_high),
        )
        sigma = log_sigma.exp()

        noise = torch.randn_like(x) * sigma
        x_noisy = x + noise
        net_input = torch.cat([x_noisy, sigma], dim=1)

        score, _ = self._score_with_log_density(net_input, create_graph=True)
        score = score[:, :-1]

        loss = torch.norm(score + noise / (sigma ** 2), dim=-1) ** 2
        lambda_factor = (sigma ** 2).squeeze(1)
        loss = (lambda_factor * loss).mean() / 2.0

        if self.beta > 0:
            clean_input = torch.cat([x, sigma], dim=1)
            _, log_density_clean = self._score_with_log_density(
                clean_input,
                create_graph=True,
            )
            reg = self.beta * (log_density_clean ** 2).mean() / 2.0
            loss = loss + reg

        return loss

    def compute_multiscale_signal(
        self,
        features: torch.Tensor,
        signal_kind: str = "log_density",
    ) -> np.ndarray:
        """Compute one of the benchmarked multi-sigma inference signals."""
        batch_size = features.shape[0]
        device = features.device
        x = self._standardize_features(features)

        if signal_kind == "log_density":
            with torch.no_grad():
                all_log_densities = []
                for sigma_val in self._get_eval_sigmas():
                    sigma_tensor = torch.full(
                        (batch_size, 1),
                        float(sigma_val),
                        device=device,
                        dtype=x.dtype,
                    )
                    net_input = torch.cat([x, sigma_tensor], dim=1)
                    log_density = self.network(net_input)
                    all_log_densities.append(log_density.cpu().numpy())

            return np.concatenate(all_log_densities, axis=1)

        if signal_kind == "score_norm":
            all_score_norms = []
            for sigma_val in self._get_eval_sigmas():
                sigma_tensor = torch.full(
                    (batch_size, 1),
                    float(sigma_val),
                    device=device,
                    dtype=x.dtype,
                )
                x_req = x.clone().requires_grad_(True)
                net_input = torch.cat([x_req, sigma_tensor], dim=1)
                score, _ = self._score_with_log_density(net_input, create_graph=False)
                score = score[:, :-1]
                lambda_factor = float(sigma_val) ** 2
                score_norm = lambda_factor * (torch.norm(score, dim=1) ** 2)
                all_score_norms.append(score_norm.unsqueeze(1).detach().cpu().numpy())

            return np.concatenate(all_score_norms, axis=1)

        raise ValueError(
            f"Unknown signal_kind={signal_kind!r}. Expected 'log_density' or 'score_norm'."
        )

    def compute_log_densities(self, features: torch.Tensor) -> np.ndarray:
        """Compute clean-feature log densities across evaluation sigmas."""
        return self.compute_multiscale_signal(features, signal_kind="log_density")

    def compute_score_norms(self, features: torch.Tensor) -> np.ndarray:
        """Compute lambda-weighted score norms across evaluation sigmas."""
        return self.compute_multiscale_signal(features, signal_kind="score_norm")

    def fit_gmm(self, train_log_densities: np.ndarray) -> None:
        """Fit a GMM on training-set log-density vectors."""
        from sklearn.mixture import GaussianMixture

        logger.info(
            "Fitting GMM: %s components on %s log-densities",
            self.gmm_components,
            train_log_densities.shape,
        )

        if train_log_densities.shape[0] < self.gmm_components:
            raise ValueError(
                f"Need at least {self.gmm_components} samples for GMM, "
                f"got {train_log_densities.shape[0]}"
            )

        train_log_densities = np.asarray(train_log_densities, dtype=np.float64)

        last_error = None
        fitted_gmm = None
        for reg_covar in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                candidate = GaussianMixture(
                    n_components=self.gmm_components,
                    covariance_type="full",
                    random_state=42,
                    max_iter=200,
                    reg_covar=reg_covar,
                )
                candidate.fit(train_log_densities)
                fitted_gmm = candidate
                break
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "GMM fit failed during checkpoint save (components=%s, reg_covar=%s): %s",
                    self.gmm_components,
                    reg_covar,
                    exc,
                )

        if fitted_gmm is None:
            raise last_error

        self._gmm = fitted_gmm
        self._train_log_densities = train_log_densities
        logger.info("GMM converged: %s", self._gmm.converged_)

    def score_anomaly(self, features: torch.Tensor) -> np.ndarray:
        """Compute anomaly scores from GMM negative log-likelihood."""
        if self._gmm is None:
            raise RuntimeError("GMM not fitted. Call fit_gmm() first.")

        log_densities = self.compute_log_densities(features)
        gmm_log_likelihood = self._gmm.score_samples(log_densities)
        return -gmm_log_likelihood

    def save_checkpoint(self, path: Path) -> None:
        """Save model, stats, and optional fitted GMM."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "sigma_low": self.sigma_low,
            "sigma_high": self.sigma_high,
            "eval_L": self.eval_L,
            "beta": self.beta,
            "gmm_components": self.gmm_components,
            "use_layernorm": self.use_layernorm,
            "train_log_densities": self._train_log_densities,
        }

        if self._legacy_eval_sigmas:
            checkpoint["legacy_eval_sigmas"] = list(self._legacy_eval_sigmas)

        if self._gmm is not None:
            gmm_path = path.parent / f"{path.stem}_gmm.pkl"
            with open(gmm_path, "wb") as handle:
                pickle.dump(self._gmm, handle)
            checkpoint["gmm_path"] = str(gmm_path)

        torch.save(checkpoint, path)
        logger.info("MULDEScorer checkpoint saved to %s", path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cpu") -> "MULDEScorer":
        """Load new-format or legacy MULDE checkpoints."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        legacy_noise_scales = checkpoint.get("noise_scales")
        sigma_low = checkpoint.get(
            "sigma_low",
            min(legacy_noise_scales) if legacy_noise_scales else 1e-3,
        )
        sigma_high = checkpoint.get(
            "sigma_high",
            max(legacy_noise_scales) if legacy_noise_scales else 1.0,
        )
        eval_L = checkpoint.get(
            "eval_L",
            len(legacy_noise_scales) if legacy_noise_scales else 16,
        )

        scorer = cls(
            feature_dim=checkpoint["feature_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 128),
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            eval_L=eval_L,
            beta=checkpoint.get("beta", 0.0) or 0.0,
            gmm_components=checkpoint.get("gmm_components", 10),
            num_layers=checkpoint.get("num_layers", 2),
            use_layernorm=checkpoint.get("use_layernorm", False),
        )

        legacy_eval_sigmas = checkpoint.get("legacy_eval_sigmas") or legacy_noise_scales
        if legacy_eval_sigmas and "sigma_low" not in checkpoint:
            scorer._legacy_eval_sigmas = list(legacy_eval_sigmas)

        missing, unexpected = scorer.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
        if missing:
            logger.warning("Checkpoint missing state keys: %s", missing)
        if unexpected:
            logger.warning("Checkpoint has unexpected state keys: %s", unexpected)

        scorer._train_log_densities = checkpoint.get("train_log_densities")

        gmm_path = checkpoint.get("gmm_path")
        gmm_candidates = []
        if gmm_path:
            gmm_candidates.append(Path(gmm_path))
            gmm_candidates.append(path.parent / Path(gmm_path).name)
        gmm_candidates.append(path.parent / f"{path.stem}_gmm.pkl")

        resolved_gmm_path = next((candidate for candidate in gmm_candidates if candidate.exists()), None)
        if resolved_gmm_path is not None:
            with open(resolved_gmm_path, "rb") as handle:
                scorer._gmm = pickle.load(handle)
            logger.info("Loaded GMM sidecar from %s", resolved_gmm_path)
        elif scorer._train_log_densities is not None:
            logger.warning(
                "GMM sidecar missing for %s; rebuilding from saved train log densities.",
                path,
            )
            scorer.fit_gmm(scorer._train_log_densities)

        scorer.to(device)
        logger.info("MULDEScorer loaded from %s", path)
        return scorer
