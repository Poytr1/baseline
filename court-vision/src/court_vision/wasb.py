"""WASB-SBDT ball detection — HRNet-based heatmap detector.

Architecture ported from nttcom/WASB-SBDT (MIT License, Copyright (c) Microsoft,
written by Bin Xiao, modified by Bowen Cheng). See
https://github.com/nttcom/WASB-SBDT and paper arXiv:2311.05237.
"""

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from court_vision.ball_tracker import BallDetection
from court_vision.device import get_device


WASB_WIDTH = 512
WASB_HEIGHT = 288
_BN_MOMENTUM = 0.1


# Hardcoded WASB tennis config (from src/configs/model/wasb.yaml upstream).
_WASB_CFG = {
    "frames_in": 3,
    "frames_out": 3,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
    },
}


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


_BLOCKS = {"BASIC": _BasicBlock, "BOTTLENECK": _Bottleneck}


class _HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=_BN_MOMENTUM),
            )

        layers = [block(self.num_inchannels[branch_index],
                        num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[i]),
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j],
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[j]),
                                nn.ReLU(True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class WASBHRNet(nn.Module):
    """HRNet backbone used by WASB-SBDT for ball heatmap prediction.

    Input:  (B, 3*frames_in, H, W) — 3 stacked RGB frames, H=288, W=512.
    Output: dict {scale: (B, frames_out, H, W)} — raw logits (no sigmoid).
    """

    def __init__(self, cfg: dict = _WASB_CFG) -> None:
        super().__init__()
        extra = cfg["MODEL"]["EXTRA"]
        self._frames_in = cfg["frames_in"]
        self._frames_out = cfg["frames_out"]
        self._out_scales = cfg["out_scales"]

        stem_strides = extra["STEM"]["STRIDES"]
        stem_inplanes = extra["STEM"]["INPLANES"]

        self.conv1 = nn.Conv2d(3 * self._frames_in, stem_inplanes,
                               kernel_size=3, stride=stem_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_inplanes, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(stem_inplanes, stem_inplanes,
                               kernel_size=3, stride=stem_strides[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(stem_inplanes, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = extra["STAGE1"]
        block1 = _BLOCKS[self.stage1_cfg["BLOCK"]]
        self.layer1 = self._make_layer(
            block1, stem_inplanes, self.stage1_cfg["NUM_CHANNELS"][0],
            self.stage1_cfg["NUM_BLOCKS"][0],
        )
        stage1_out_channels = block1.expansion * self.stage1_cfg["NUM_CHANNELS"][0]

        self.stage2_cfg = extra["STAGE2"]
        block2 = _BLOCKS[self.stage2_cfg["BLOCK"]]
        num_channels_2 = [c * block2.expansion for c in self.stage2_cfg["NUM_CHANNELS"]]
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels_2)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels_2)

        self.stage3_cfg = extra["STAGE3"]
        block3 = _BLOCKS[self.stage3_cfg["BLOCK"]]
        num_channels_3 = [c * block3.expansion for c in self.stage3_cfg["NUM_CHANNELS"]]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels_3)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels_3)

        self.stage4_cfg = extra["STAGE4"]
        block4 = _BLOCKS[self.stage4_cfg["BLOCK"]]
        num_channels_4 = [c * block4.expansion for c in self.stage4_cfg["NUM_CHANNELS"]]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels_4)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels_4, multi_scale_output=True,
        )

        deconv_cfg = extra["DECONV"]
        self.num_deconvs = deconv_cfg["NUM_DECONVS"]
        self.deconv_layers = nn.ModuleList()  # empty for WASB (NUM_DECONVS=0)

        kernel_size = extra["FINAL_CONV_KERNEL"]
        self.final_layers = nn.ModuleList([
            nn.Conv2d(pre_stage_channels[scale], self._frames_out, kernel_size=kernel_size)
            for scale in self._out_scales
        ])

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=_BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        num_branches_cur = len(num_channels_cur)
        num_branches_pre = len(num_channels_pre)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i],
                                  3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i], momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre[-1]
                    outchannels = num_channels_cur[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_cfg, num_inchannels, multi_scale_output=True):
        num_modules = layer_cfg["NUM_MODULES"]
        num_branches = layer_cfg["NUM_BRANCHES"]
        num_blocks = layer_cfg["NUM_BLOCKS"]
        num_channels = layer_cfg["NUM_CHANNELS"]
        block = _BLOCKS[layer_cfg["BLOCK"]]
        fuse_method = layer_cfg["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = not (not multi_scale_output and i == num_modules - 1)
            modules.append(_HighResolutionModule(
                num_branches, block, num_blocks, num_inchannels, num_channels,
                fuse_method, reset_multi_scale_output,
            ))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        y_out = {}
        for idx, scale in enumerate(self._out_scales):
            y_out[scale] = self.final_layers[idx](y_list[scale])
        return y_out


def extract_ball_candidates(
    heatmap: np.ndarray,
    original_width: int,
    original_height: int,
    confidence_threshold: float = 0.5,
    max_candidates: int = 5,
) -> list[tuple[float, float, float]]:
    """Every heatmap blob above threshold, strongest first.

    Same connected-component / weighted-centroid extraction as
    :func:`_extract_ball_position_weighted`, but all components are kept
    (up to ``max_candidates``, ranked by their summed heat) and each gets
    its own peak value as confidence. On a court with loose balls lying
    around the strongest blob is often one of them; the trajectory stage
    picks the motion-consistent path through these candidates instead.

    Returns:
        List of (x, y, confidence) in original frame coordinates.
    """
    if float(np.max(heatmap)) < confidence_threshold:
        return []
    binary = (heatmap > confidence_threshold).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary)
    hm_h, hm_w = heatmap.shape
    found = []
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        weights = heatmap[ys, xs]
        total = float(weights.sum())
        if total <= 0.0:
            continue
        x = float(np.sum(xs * weights) / total) / hm_w * original_width
        y = float(np.sum(ys * weights) / total) / hm_h * original_height
        found.append((x, y, float(weights.max()), total))
    found.sort(key=lambda c: -c[3])
    return [(x, y, peak) for x, y, peak, _ in found[:max_candidates]]


def _extract_ball_position_weighted(
    heatmap: np.ndarray,
    original_width: int,
    original_height: int,
    confidence_threshold: float = 0.5,
) -> tuple[float, float, float] | None:
    """Connected-component + weighted centroid peak extraction.

    Mirrors WASB's `concomp` postprocessor with `use_hm_weight=True`:
    threshold the heatmap, find connected components, and within each
    component compute a heatmap-weighted centroid. The highest-scoring
    component wins.

    Args:
        heatmap: 2D array (H, W), values in [0, 1] (already sigmoided).
        original_width: Original frame width in pixels.
        original_height: Original frame height in pixels.
        confidence_threshold: Min peak value to accept.

    Returns:
        (x, y, confidence) in original frame coordinates, or None.
    """
    peak_value = float(np.max(heatmap))
    if peak_value < confidence_threshold:
        return None

    binary = (heatmap > confidence_threshold).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary)

    best_score = -1.0
    best_x, best_y = 0.0, 0.0
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        weights = heatmap[ys, xs]
        total = float(weights.sum())
        if total <= 0.0:
            continue
        x = float(np.sum(xs * weights) / total)
        y = float(np.sum(ys * weights) / total)
        if total > best_score:
            best_score = total
            best_x, best_y = x, y

    if best_score < 0.0:
        return None

    hm_h, hm_w = heatmap.shape
    x_scaled = best_x / hm_w * original_width
    y_scaled = best_y / hm_h * original_height
    return (x_scaled, y_scaled, peak_value)


logger = logging.getLogger(__name__)

_WEIGHTS_GDRIVE_ID = "14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z"
_CACHE_DIR = Path.home() / ".cache" / "court-vision" / "models"
_WEIGHTS_FILENAME = "wasb_tennis.pth.tar"


def _download_wasb_weights() -> Path:
    """Download pretrained WASB tennis weights via gdown if not cached."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = _CACHE_DIR / _WEIGHTS_FILENAME

    if weights_path.exists():
        return weights_path

    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required to download WASB weights. "
            "Install with: pip install gdown",
        ) from e

    logger.info("Downloading WASB tennis weights to %s...", weights_path)
    gdown.download(id=_WEIGHTS_GDRIVE_ID, output=str(weights_path), quiet=False)
    logger.info("WASB weights downloaded.")
    return weights_path


def _strip_module_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix left over from DataParallel-wrapped checkpoints."""
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


@lru_cache(maxsize=1)
def _get_wasb_model() -> WASBHRNet:
    """Load the WASB HRNet model (cached singleton)."""
    weights_path = _download_wasb_weights()
    device = get_device()

    model = WASBHRNet()

    if weights_path.exists():
        try:
            checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            state_dict = _strip_module_prefix(state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("WASB weights: %d missing keys (e.g. %s)",
                               len(missing), missing[:3])
            if unexpected:
                logger.warning("WASB weights: %d unexpected keys (e.g. %s)",
                               len(unexpected), unexpected[:3])
            logger.info("WASB weights loaded from %s", weights_path)
        except Exception as e:
            logger.warning("Failed to load WASB weights: %s. Using random init.", e)

    model = model.to(device)
    model.eval()
    return model


def _wasb_heatmap(frames: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Run WASB on a 3-frame window; the current frame's sigmoided heatmap
    plus the original frame size."""
    if len(frames) != 3:
        raise ValueError(f"detect_ball_wasb requires exactly 3 frames, got {len(frames)}")

    original_height, original_width = frames[0].shape[:2]
    device = get_device()
    model = _get_wasb_model()

    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (WASB_WIDTH, WASB_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed.append(rgb)

    concatenated = np.concatenate(processed, axis=2)  # (H, W, 9)
    tensor = torch.from_numpy(concatenated).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output_dict = model(tensor)
    logits = output_dict[0]  # (1, 3, 288, 512)
    heatmap = torch.sigmoid(logits[0, 2]).cpu().numpy()  # current frame channel
    return heatmap, original_width, original_height


def detect_ball_candidates_wasb(
    frames: list[np.ndarray],
    frame_index: int,
    confidence_threshold: float = 0.5,
    max_candidates: int = 5,
) -> list[BallDetection]:
    """Like :func:`detect_ball_wasb` but every blob above threshold, strongest
    first (see :func:`extract_ball_candidates`)."""
    heatmap, w, h = _wasb_heatmap(frames)
    return [
        BallDetection(frame_index=frame_index, x=x, y=y, confidence=c)
        for x, y, c in extract_ball_candidates(heatmap, w, h, confidence_threshold, max_candidates)
    ]


def detect_ball_wasb(
    frames: list[np.ndarray],
    frame_index: int,
    confidence_threshold: float = 0.5,
) -> BallDetection | None:
    """Detect the tennis ball using WASB (HRNet heatmap detector).

    Takes exactly 3 consecutive BGR frames, runs WASB inference, and
    returns the ball position extracted via connected-component +
    weighted centroid from the heatmap for the current (last) frame.

    Args:
        frames: Exactly 3 BGR frames (any resolution; resized to 512x288).
        frame_index: Frame index for the detection result.
        confidence_threshold: Min heatmap peak value to accept.

    Returns:
        BallDetection in original frame coords, or None.

    Raises:
        ValueError: If not exactly 3 frames provided.
    """
    heatmap, original_width, original_height = _wasb_heatmap(frames)
    result = _extract_ball_position_weighted(
        heatmap, original_width, original_height, confidence_threshold,
    )
    if result is None:
        return None

    x, y, confidence = result
    return BallDetection(
        frame_index=frame_index,
        x=x,
        y=y,
        confidence=confidence,
    )
