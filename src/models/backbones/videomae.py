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
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Monkey patching utility to capture intermediate ViT attention weights
# ──────────────────────────────────────────────────────────────────────
def patch_attention_forward(attn_module):
    def new_forward(x):
        B, N, C = x.shape
        qkv_bias = None
        if attn_module.q_bias is not None:
            qkv_bias = torch.cat(
                (attn_module.q_bias,
                 torch.zeros_like(attn_module.v_bias, requires_grad=False), 
                 attn_module.v_bias)
            )
        qkv = F.linear(input=x, weight=attn_module.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * attn_module.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        # Save attention weights
        attn_module.captured_attn = attn
        
        attn_dropped = attn_module.attn_drop(attn)
        x_out = (attn_dropped @ v).transpose(1, 2).reshape(B, N, -1)
        x_out = attn_module.proj(x_out)
        x_out = attn_module.proj_drop(x_out)
        return x_out
        
    attn_module.forward = new_forward

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
        self.dtype = torch.float16 if "cuda" in device else torch.float32
        self.model_name = model_name

        logger.info(f"Loading VideoMAEv2 backbone: {model_name} on {device} ({'FP16' if self.dtype == torch.float16 else 'FP32'})")

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
        
        # Load weights directly to target device in the matching precision
        model = model.to(device=self.device, dtype=self.dtype)
        
        state_dict = load_safetensors(weights_path, device=self.device)
        
        # Convert state_dict tensors to target precision to match model
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].to(dtype=self.dtype)
            
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"VideoMAEv2: {len(missing)} missing weight keys")
        if unexpected:
            logger.warning(f"VideoMAEv2: {len(unexpected)} unexpected weight keys")

        self.model = model
        self.model.eval()

        # ── Probe: detect true hidden_size and output format ───────────────
        # VideoMAEv2Config uses 'embed_dim' not 'hidden_size'.
        # Also detect whether the model returns a raw tensor or a ModelOutput,
        # and whether it accepts 'pixel_values' keyword or positional args.
        # Doing this once here avoids per-batch conditionals later.
        with torch.no_grad():
            _probe = torch.zeros(1, 3, CLIP_LENGTH, FRAME_SIZE, FRAME_SIZE,
                                 dtype=self.dtype, device=self.device)
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

        # Monkey-patch final attention blocks (9, 10, 11) for spatiotemporal rollout explainability
        self.patched_attn_modules = []
        for i in [9, 10, 11]:
            try:
                module = self.model.get_submodule(f"model.blocks.{i}.attn")
                patch_attention_forward(module)
                self.patched_attn_modules.append(module)
                logger.info(f"Successfully monkey-patched layer {i} attention block.")
            except Exception as e:
                logger.warning(f"Could not patch layer {i} attention block: {e}")

        # ONNX acceleration and dynamic compile setup
        self.use_onnx = False
        if device == "cuda":
            try:
                import onnxruntime as ort
                self.onnx_path = Path(os.environ.get("HF_HOME", "/cache/huggingface")) / "videomae_base_fp16.onnx"
                self.onnx_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not self.onnx_path.exists():
                    logger.info(f"ONNX backbone not found at {self.onnx_path}. Initiating automatic export...")
                    self._export_to_onnx(self.model, self.onnx_path, device)
                
                logger.info(f"Loading VideoMAE backbone via ONNX Runtime from {self.onnx_path}")
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                self.use_onnx = True
            except Exception as e:
                logger.warning(f"Failed to load ONNX backbone or onnxruntime not available. Falling back to native PyTorch: {e}")

        logger.info(
            f"VideoMAEv2 loaded: hidden_size={self.hidden_size}, "
            f"output_mode={self._output_mode}, "
            f"use_onnx={self.use_onnx}, "
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

            if self.use_onnx:
                ort_inputs = {self.ort_session.get_inputs()[0].name: batch.numpy()}
                ort_outputs = self.ort_session.run(None, ort_inputs)
                pooled_np = ort_outputs[0].astype(np.float16)
                all_features.append(pooled_np)
            else:
                batch = batch.to(device=self.device, dtype=self.dtype, non_blocking=True)
                if "cuda" in self.device:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        if self._kw == "pixel_values":
                            outputs = self.model(pixel_values=batch)
                        else:
                            outputs = self.model(batch)
                else:
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
        """Extract VideoMAEv2 features directly from in-memory RGB frames."""
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

            if self.use_onnx:
                ort_inputs = {self.ort_session.get_inputs()[0].name: batch.numpy()}
                ort_outputs = self.ort_session.run(None, ort_inputs)
                pooled_np = ort_outputs[0].astype(np.float16)
                all_features.append(pooled_np)
            else:
                batch = batch.to(device=self.device, dtype=self.dtype, non_blocking=True)
                if "cuda" in self.device:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        if self._kw == "pixel_values":
                            outputs = self.model(pixel_values=batch)
                        else:
                            outputs = self.model(batch)
                else:
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

    def _export_to_onnx(self, model, onnx_path: Path, device: str):
        import onnx
        from onnxconverter_common import float16
        
        logger.info("Building ONNX export wrapper...")
        class ONNXVideoMAEWrapper(torch.nn.Module):
            def __init__(self, inner_model, output_mode):
                super().__init__()
                self.inner_model = inner_model
                self.output_mode = output_mode

            def forward(self, pixel_values):
                outputs = self.inner_model(pixel_values)
                if isinstance(outputs, torch.Tensor):
                    raw = outputs
                elif hasattr(outputs, "last_hidden_state"):
                    raw = outputs.last_hidden_state
                elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                else:
                    raw = outputs[0]
                return raw.mean(dim=1)

        wrapper = ONNXVideoMAEWrapper(model, self._output_mode)
        wrapper.eval()
        
        # Export in float32 for stability
        dummy_input = torch.randn(1, 3, CLIP_LENGTH, FRAME_SIZE, FRAME_SIZE, dtype=torch.float32, device=device)
        wrapper = wrapper.to(device=device, dtype=torch.float32)
        wrapper.inner_model = wrapper.inner_model.to(dtype=torch.float32)
        
        tmp_path = onnx_path.with_suffix(".tmp.onnx")
        logger.info(f"Tracing PyTorch model and saving temporary FP32 ONNX to {tmp_path}...")
        
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_input,
                str(tmp_path),
                input_names=["pixel_values"],
                output_names=["features"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "features": {0: "batch_size"}
                },
                opset_version=17
            )
            
        logger.info("Converting ONNX model to FP16 half precision...")
        model_fp32 = onnx.load(str(tmp_path))
        model_fp16 = float16.convert_float16(model_fp32, keep_ids_limits=True)
        onnx.save(model_fp16, str(onnx_path))
        
        # Clean up temporary FP32 file
        if tmp_path.exists():
            tmp_path.unlink()
            
        # Restore PyTorch model to original dtype
        wrapper.inner_model = wrapper.inner_model.to(dtype=self.dtype)
        logger.info("ONNX FP16 export completed successfully.")

    def generate_explainability_heatmap(
        self,
        clip_frames: List[np.ndarray],
        scorer: torch.nn.Module,
        target_device: str = "cuda"
    ) -> np.ndarray:
        """Compute gradient-weighted multi-layer attention rollout for 16-frame clip."""
        # Prepare clip tensor
        clip_array = np.stack(clip_frames).transpose(0, 3, 1, 2)  # [16, 3, 224, 224]
        clip_tensor = torch.from_numpy(clip_array).unsqueeze(0).float().div_(255.0)
        mean_t = torch.tensor(self.image_mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(self.image_std, dtype=torch.float32).view(3, 1, 1)
        clip_tensor = (clip_tensor - mean_t) / std_t
        clip_tensor = clip_tensor.permute(0, 2, 1, 3, 4)  # [1, 3, 16, 224, 224]
        
        device = "cuda" if torch.cuda.is_available() and "cuda" in target_device else "cpu"
        dtype = torch.float16 if "cuda" in device else torch.float32
        clip_tensor = clip_tensor.to(device, dtype=dtype).requires_grad_(True)
        
        self.model.zero_grad()
        scorer.zero_grad()
        
        # Enable gradient computation locally for the explainability backward pass
        with torch.enable_grad():
            if self._kw == "pixel_values":
                outputs = self.model(pixel_values=clip_tensor)
            else:
                outputs = self.model(clip_tensor)
                
            if self._output_mode == "tensor":
                raw = outputs
                pooled = raw.mean(dim=1) if raw.dim() == 3 else raw
            elif self._output_mode == "pooler_output":
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state.mean(dim=1)
                
            # Retain gradients of intermediate attention maps
            attn_weights = {}
            for i in [9, 10, 11]:
                module = self.model.get_submodule(f"model.blocks.{i}.attn")
                if hasattr(module, "captured_attn") and module.captured_attn is not None:
                    attn_weights[i] = module.captured_attn
                    attn_weights[i].retain_grad()
                    
            # Scorer forward
            x = scorer._standardize_features(pooled)
            score = 0.0
            for sigma_val in scorer._get_eval_sigmas():
                sigma_tensor = torch.full((1, 1), float(sigma_val), device=device, dtype=x.dtype)
                net_input = torch.cat([x, sigma_tensor], dim=1)
                log_density = scorer.network(net_input)
                score -= log_density.sum()
                
            score.backward()
            
        # Compute rollout map
        rollout_map = None
        num_tokens = 1568  # 8 * 14 * 14
        identity = torch.eye(num_tokens, device=device, dtype=dtype)
        
        for i in [9, 10, 11]:
            if i not in attn_weights or attn_weights[i].grad is None:
                continue
            
            attn = attn_weights[i]  # [1, 12, 1568, 1568]
            grad = attn.grad  # [1, 12, 1568, 1568]
            
            weights = torch.clamp(grad, min=0)
            m_l = (weights * attn).sum(dim=1).squeeze(0)  # [1568, 1568]
            
            max_val = m_l.max()
            if max_val > 0:
                m_l = m_l / max_val
                
            m_l_prime = 0.5 * identity + 0.5 * m_l
            
            if rollout_map is None:
                rollout_map = m_l_prime
            else:
                rollout_map = torch.matmul(m_l_prime, rollout_map)
                
        if rollout_map is None:
            rollout_map = identity
            
        h_j = rollout_map.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        h_j = h_j.reshape(8, 14, 14)
        heatmap_14x14 = h_j.mean(axis=0)
        
        max_val = heatmap_14x14.max()
        if max_val > 0:
            heatmap_14x14 = heatmap_14x14 / max_val
            
        return heatmap_14x14
