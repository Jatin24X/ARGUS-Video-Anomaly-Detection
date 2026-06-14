"""VideoMAEv2-Base feature extractor — Stream A.

Source: architecture_detail.md Gap 5.3 (Stream A), lines 472-479
Model: OpenGVLab/VideoMAEv2-Base (frozen, CVPR 2023)
Input: 16-frame clips, 224×224, temporal sampling stride 4, sliding window
Output: 768-dim mean-pooled embedding per clip
Saved as: {video_name}.npy — shape [num_clips, 768], dtype float16

Clip construction:
  - CLIP_LENGTH = 16 frames per clip (model input)
  - TEMPORAL_STRIDE = 4: within each clip, sample every 4th raw frame
    so each clip reads raw frames [start, start+4, start+8, ..., start+60]
  - Clip START positions slide by TEMPORAL_STRIDE (step=4 raw frames)
    producing overlapping windows: high temporal resolution
  - num_clips = max(1, (num_frames - CLIP_LENGTH) // TEMPORAL_STRIDE + 1)
  - Example: 131 frames → (131-16)//4 + 1 = 29 clips

Optimized for NVIDIA L4 (24GB VRAM), 8 vCPUs, 32GB RAM on Lightning AI.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Global backend tuning for L4 Tensor Cores
# ──────────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ──────────────────────────────────────────────────────────────────────
# Constants (from architecture_detail.md Gap 5.3)
# ──────────────────────────────────────────────────────────────────────
CLIP_LENGTH = 16          # Frames per clip (model's num_frames config)
TEMPORAL_STRIDE = 4       # Sample every 4th raw frame within each clip
                          # Also used as the clip-start step (sliding window)
FRAME_SIZE = 224          # Model input resolution


# ──────────────────────────────────────────────────────────────────────
# Dataset: Clip-level loading with OpenCV + model-config normalization
# ──────────────────────────────────────────────────────────────────────
class _VideoMAEClipDataset(Dataset):
    """Loads 16-frame clips for a single video with temporal stride 4.

    Each clip spans 64 raw frames, sampling every 4th frame.
    Clips are OVERLAPPING (sliding window, stride=4 between clip starts).
    Uses OpenCV C++ decoder for fast image loading.
    """

    def __init__(
        self,
        frame_paths: List[Path],
        image_mean: List[float],
        image_std: List[float],
    ):
        self.frame_paths = frame_paths
        self.num_frames = len(frame_paths)

        # Clip start positions: slide by TEMPORAL_STRIDE (step=4 raw frames)
        # This produces overlapping windows with high temporal resolution.
        # E.g. 131 frames -> (131-16)//4 + 1 = 29 clips.
        if self.num_frames >= CLIP_LENGTH:
            self.clip_starts = list(
                range(0, self.num_frames - CLIP_LENGTH + 1, TEMPORAL_STRIDE)
            )
        else:
            self.clip_starts = [0]  # short video: one clip from frame 0

        # Normalization tensors — loaded from VideoMAEImageProcessor config
        self.mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.clip_starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self.clip_starts[idx]

        frames = []
        for i in range(CLIP_LENGTH):
            raw_idx = start + i * TEMPORAL_STRIDE
            # Clamp to last frame if we exceed video length
            raw_idx = min(raw_idx, self.num_frames - 1)

            img = cv2.imread(str(self.frame_paths[raw_idx]))
            if img is None:
                img = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR
                )
            frames.append(img)

        # Stack: [16, 224, 224, 3] → transpose → [16, 3, 224, 224]
        clip_array = np.stack(frames).transpose(0, 3, 1, 2)

        # Convert to float, normalize with model-config values
        tensor = torch.from_numpy(clip_array).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std  # Broadcasts [3,1,1] over [16,3,224,224]

        return tensor  # [16, 3, 224, 224]


class _VideoMAEInMemoryClipDataset(Dataset):
    """Loads clips from already resized RGB frames kept in memory."""

    def __init__(
        self,
        frames_rgb: List[np.ndarray],
        image_mean: List[float],
        image_std: List[float],
    ):
        self.frames_rgb = frames_rgb
        self.num_frames = len(frames_rgb)

        if self.num_frames >= CLIP_LENGTH:
            self.clip_starts = list(
                range(0, self.num_frames - CLIP_LENGTH + 1, TEMPORAL_STRIDE)
            )
        else:
            self.clip_starts = [0]

        self.mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.clip_starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self.clip_starts[idx]
        frames = []
        for i in range(CLIP_LENGTH):
            raw_idx = start + i * TEMPORAL_STRIDE
            raw_idx = min(raw_idx, self.num_frames - 1)
            frames.append(self.frames_rgb[raw_idx])

        clip_array = np.stack(frames).transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(clip_array).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor


# ──────────────────────────────────────────────────────────────────────
# Feature Extractor
# ──────────────────────────────────────────────────────────────────────
class VideoMAEFeatureExtractor:
    """Extracts 768-dim spatiotemporal features from VideoMAEv2-Base.

    Architecture: architecture_detail.md line 88 — "VideoMAEv2-Base (frozen)"
    Model: OpenGVLab/VideoMAEv2-Base (CVPR 2023, dual masking pre-training)

    CRITICAL API NOTE (from official HuggingFace example):
    VideoMAEv2 expects pixel_values in shape [B, C, T, H, W].
    The VideoMAEImageProcessor outputs [B, T, C, H, W].
    We must apply .permute(0, 2, 1, 3, 4) before feeding to the model.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/VideoMAEv2-Base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_name = model_name

        logger.info(f"Loading VideoMAEv2 backbone: {model_name} on {device} (FP16)")

        from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors

        # Load processor for normalization values — don't hardcode ImageNet stats
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.image_mean = list(self.processor.image_mean)
        self.image_std = list(self.processor.image_std)
        logger.info(f"VideoMAE normalization: mean={self.image_mean}, std={self.image_std}")

        # ── CRITICAL FIX: from_config + manual weight loading ──────────────
        # from_pretrained uses init_empty_weights() from accelerate (meta device
        # lazy loading). The custom modeling_videomaev2.py calls
        # torch.linspace(...).item() during __init__, which crashes:
        #   "Tensor.item() cannot be called on meta tensors"
        # low_cpu_mem_usage=False does NOT fully prevent this in transformers>=4.38
        # when accelerate is installed.
        #
        # Solution: from_config() creates the model on real CPU tensors (no meta
        # device), then we load pretrained weights manually from the cached
        # safetensors file — completely bypassing the meta tensor code path.
        # ───────────────────────────────────────────────────────────────────
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        logger.info("Instantiating VideoMAEv2 model on CPU (from_config)...")
        model = AutoModel.from_config(config, trust_remote_code=True)

        logger.info("Loading pretrained weights from safetensors cache...")
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        state_dict = load_safetensors(weights_path, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"VideoMAEv2: {len(missing)} missing weight keys")
        if unexpected:
            logger.warning(f"VideoMAEv2: {len(unexpected)} unexpected weight keys")

        self.model = model.half().to(device)
        self.model.eval()

        # ── Probe: detect true hidden_size and output format ───────────────
        # VideoMAEv2Config uses 'embed_dim' not 'hidden_size'.
        # Also detect whether the model returns a raw tensor or a ModelOutput,
        # and whether it accepts 'pixel_values' keyword or positional args.
        # Doing this once here avoids per-batch conditionals later.
        with torch.no_grad():
            _probe = torch.zeros(1, 3, CLIP_LENGTH, FRAME_SIZE, FRAME_SIZE,
                                 dtype=torch.float16, device=device)
            try:
                _out = self.model(pixel_values=_probe)
                self._kw = "pixel_values"
            except TypeError:
                _out = self.model(_probe)
                self._kw = None  # positional only

            if isinstance(_out, torch.Tensor):
                # Returns raw tensor: either [B, N, C] (patch tokens) or [B, C]
                self._output_mode = "tensor"
                self.hidden_size = _out.shape[-1]
            elif hasattr(_out, "last_hidden_state"):
                self._output_mode = "last_hidden_state"
                self.hidden_size = _out.last_hidden_state.shape[-1]
            elif hasattr(_out, "pooler_output") and _out.pooler_output is not None:
                self._output_mode = "pooler_output"
                self.hidden_size = _out.pooler_output.shape[-1]
            else:
                # Last resort: read from config attribute (try several names)
                self._output_mode = "last_hidden_state"
                self.hidden_size = (
                    getattr(config, "hidden_size", None)
                    or getattr(config, "embed_dim", None)
                    or 768
                )
            del _probe, _out

        logger.info(
            f"VideoMAEv2 loaded: hidden_size={self.hidden_size}, "
            f"output_mode={self._output_mode}, "
            f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
        )

    @torch.inference_mode()
    def extract_single_video(
        self,
        frame_paths: List[Path],
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> np.ndarray:
        """Extract VideoMAEv2 features for all clips of one video.

        Args:
            frame_paths: Sorted list of all frame image paths for this video.
            batch_size: Clips per GPU batch. 16 is safe for L4 24GB with
                        VideoMAEv2-Base (each clip is 16×3×224×224 in FP16).
            num_workers: DataLoader workers for parallel image decoding.

        Returns:
            np.ndarray of shape [num_clips, 768], dtype float16.
            num_clips = max(1, (total_frames - CLIP_LENGTH) // TEMPORAL_STRIDE + 1)
        """
        if not frame_paths:
            return np.empty((0, self.hidden_size), dtype=np.float16)

        dataset = _VideoMAEClipDataset(
            frame_paths,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device != "cpu"),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False,
            drop_last=False,
        )

        all_features = []

        for batch in dataloader:
            # batch shape: [B, 16, 3, 224, 224]  (from Dataset)
            # CRITICAL: VideoMAEv2 expects [B, C, T, H, W] = [B, 3, 16, 224, 224]
            batch = batch.permute(0, 2, 1, 3, 4)
            batch = batch.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if "cuda" in self.device else "cpu",
                dtype=torch.float16,
            ):
                if self._kw == "pixel_values":
                    outputs = self.model(pixel_values=batch)
                else:
                    outputs = self.model(batch)

            # Extract patch token embeddings and mean-pool → [B, hidden_size]
            if self._output_mode == "tensor":
                raw = outputs
                pooled = raw.mean(dim=1) if raw.dim() == 3 else raw
            elif self._output_mode == "pooler_output":
                pooled = outputs.pooler_output
            else:  # last_hidden_state
                pooled = outputs.last_hidden_state.mean(dim=1)

            all_features.append(pooled.cpu().numpy().astype(np.float16))

        if not all_features:
            return np.empty((0, self.hidden_size), dtype=np.float16)

        return np.concatenate(all_features, axis=0)  # [num_clips, hidden_size]

    @torch.inference_mode()
    def extract_from_frames(
        self,
        frames_rgb: List[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Extract VideoMAEv2 features directly from in-memory RGB frames.

        This avoids the demo-time temp-folder roundtrip of writing JPEGs and
        reading them back through OpenCV, which is noticeably slower on local
        machines for short interactive runs.
        """
        if not frames_rgb:
            return np.empty((0, self.hidden_size), dtype=np.float16)

        dataset = _VideoMAEInMemoryClipDataset(
            frames_rgb,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device != "cpu"),
            drop_last=False,
        )

        all_features = []

        for batch in dataloader:
            batch = batch.permute(0, 2, 1, 3, 4)
            batch = batch.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if "cuda" in self.device else "cpu",
                dtype=torch.float16,
            ):
                if self._kw == "pixel_values":
                    outputs = self.model(pixel_values=batch)
                else:
                    outputs = self.model(batch)

            if self._output_mode == "tensor":
                raw = outputs
                pooled = raw.mean(dim=1) if raw.dim() == 3 else raw
            elif self._output_mode == "pooler_output":
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state.mean(dim=1)

            all_features.append(pooled.cpu().numpy().astype(np.float16))

        if not all_features:
            return np.empty((0, self.hidden_size), dtype=np.float16)

        return np.concatenate(all_features, axis=0)
