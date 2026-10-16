"""Author: Tamjit Kumar
NetID: tkumar70
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Temporal Pointwise Convolution (TPC) model and TPCBlock architecture
             for ICU length of stay prediction and multitask mortality prediction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.models import BaseModel

if TYPE_CHECKING:
    from pyhealth.datasets import SampleDataset

logger = logging.getLogger(__name__)


class TPCBlock(nn.Module):
    """A single Temporal Pointwise Convolution (TPC) block.

    Implements Equations (1)-(5) and Figure 3 from:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit' (Rocheteau et al., ACM CHIL 2021).

    Args:
        layer_idx (int): 1-indexed layer number (n).
        num_features (int): Number of initial time-series features (F).
        point_dim (int): Pointwise output channels (Z).
        temp_dim (int): Temporal output channels per feature (Y).
        static_dim (int): Number of static features (S).
        kernel_size (int): Filter size (k).
        dropout (float): Dropout probability.
        use_decay (bool): Whether decay indicators are used. Default: True.

    Examples:
        >>> import torch
        >>> from pyhealth.models.tpc import TPCBlock
        >>> block = TPCBlock(
        ...     layer_idx=1,
        ...     num_features=12,
        ...     point_dim=16,
        ...     temp_dim=8,
        ...     static_dim=5,
        ... )
        >>> x_raw = torch.randn(2, 12, 24)
        >>> x_decay = torch.rand(2, 12, 24)
        >>> h = torch.stack([x_raw, x_decay], dim=2)
        >>> s_expanded = torch.randn(2, 5, 24)
        >>> h_next, point_out = block(
        ...     h, x_raw, x_decay, s_expanded, prev_pointwise=[]
        ... )
        >>> h_next.shape
        torch.Size([2, 28, 9, 24])
    """

    def __init__(
        self,
        layer_idx: int,
        num_features: int,
        point_dim: int,
        temp_dim: int,
        static_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_decay: bool = True,
    ):
        super().__init__()
        self.n = layer_idx
        self.k = kernel_size
        self.d = layer_idx  # Dilation increments by 1 with each layer (d = n)
        self.pad = self.d * (self.k - 1)  # Causal left-sided padding
        self.use_decay = use_decay

        self.R_n = num_features + point_dim * (self.n - 1)  # Number of features
        if self.n == 1:
            self.C_n = 2 if self.use_decay else 1
        else:
            self.C_n = temp_dim + 1
        self.temp_dim = temp_dim
        self.point_dim = point_dim
        self.static_dim = static_dim

        # 1. Temporal Convolution (Independent filter per feature across time)
        self.temporal_conv = nn.Conv1d(
            in_channels=self.R_n * self.C_n,
            out_channels=self.R_n * self.temp_dim,
            kernel_size=self.k,
            dilation=self.d,
            groups=self.R_n,  # No weight sharing across features
            bias=True,
        )
        self.temp_bn = nn.BatchNorm1d(self.R_n * self.temp_dim)
        self.temp_dropout = nn.Dropout(dropout)

        # 2. Pointwise Convolution (1x1 cross-feature interaction)
        decay_dim = num_features if self.use_decay else 0
        self.P_n = (self.R_n * self.C_n) + self.static_dim + decay_dim
        self.pointwise_conv = nn.Conv1d(
            in_channels=self.P_n,
            out_channels=self.point_dim,
            kernel_size=1,
            bias=True,
        )
        self.point_bn = nn.BatchNorm1d(self.point_dim)
        self.point_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        x_raw: torch.Tensor,
        x_decay: torch.Tensor | None,
        s_expanded: torch.Tensor | None,
        prev_pointwise: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of a single TPC block.

        Args:
            h: Input tensor of shape (batch, R_n, C_n, seq_len).
            x_raw: Raw time series of shape (batch, F, seq_len).
            x_decay: Decay indicators of shape (batch, F, seq_len) or None.
            s_expanded: Static features of shape (batch, S, seq_len) or None.
            prev_pointwise: List of pointwise outputs from previous layers.

        Returns:
            h_next: Output tensor of shape (batch, R_{n+1}, Y+1, seq_len).
            point_out_channel: Pointwise output of shape (batch, Z, 1, seq_len).
        """
        B, _, _, T = h.shape

        # --- A. Temporal Convolution Branch ---
        h_flat = h.view(B, self.R_n * self.C_n, T)
        # Left-sided causal padding ensures no information leakage from future
        h_pad = F.pad(h_flat, (self.pad, 0))
        temp_out = self.temporal_conv(h_pad)
        temp_out = self.temp_dropout(self.temp_bn(temp_out))
        temp_out = temp_out.view(B, self.R_n, self.temp_dim, T)

        # --- B. Pointwise Convolution Branch ---
        point_in_list = [h_flat]
        if s_expanded is not None and self.static_dim > 0:
            point_in_list.append(s_expanded)
        if self.use_decay and x_decay is not None:
            point_in_list.append(x_decay)
        point_in = torch.cat(point_in_list, dim=1)

        point_out = self.pointwise_conv(point_in)
        point_out = self.point_dropout(self.point_bn(point_out))  # (B, Z, T)

        # --- C. Skip Connections (Eq. 4) ---
        if self.n == 1:
            r_n = torch.cat(
                [temp_out, x_raw.unsqueeze(2)], dim=2
            )  # (B, F, Y+1, T)
        else:
            skips = torch.cat([x_raw.unsqueeze(2)] + prev_pointwise, dim=1)
            r_n = torch.cat([temp_out, skips], dim=2)  # (B, R_n, Y+1, T)

        # --- D. Parallel Feature Concatenation & Activation (Eq. 5) ---
        # Broadcast pointwise output across channel dimension (Y + 1 times)
        point_broadcast = point_out.unsqueeze(2).repeat(
            1, 1, self.temp_dim + 1, 1
        )
        h_next = F.relu(
            torch.cat([r_n, point_broadcast], dim=1)
        )  # (B, R_n + Z, Y+1, T)

        return h_next, point_out.unsqueeze(2)


class TPC(BaseModel):
    """Temporal Pointwise Convolution (TPC) model for ICU LoS prediction.

    Adapted from the architecture described in:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit'
    (Emma Rocheteau, Pietro Liò, & Stephanie Hyland, ACM CHIL 2021).

    Args:
        dataset: Optional PyHealth SampleDataset. If provided and feature dims
            are not specified, continuous_dim and static_dim are inferred.
        feature_keys: List of input feature keys (e.g. ['vitals', 'static']).
        label_key: Target key (defaults to 'los').
        mode: Prediction mode ('regression').
        continuous_dim (int | None): Number of time series features (F).
            Inferred from dataset if None, or defaults to 12.
        static_dim (int | None): Number of static baseline features (S).
            Inferred from dataset if None, or defaults to 0.
        n_layers (int): Number of stacked TPC blocks (N). Default: 3.
        temp_dim (int): Temporal output channels per feature (Y). Default: 8.
        point_dim (int): Pointwise output channels (Z). Default: 16.
        kernel_size (int): Temporal filter size (k). Default: 3.
        hidden_dim (int): Hidden dimension for prediction head. Default: 64.
        dropout (float): Dropout probability. Default: 0.1.
        loss_fn (str): Loss function to use ('msle' or 'mse'). Default: 'msle'.
        min_time (int): Warmup time index before evaluation begins (t >= min_time).
            Default: 4 (corresponding to 5 hours elapsed since ICU admission).
        floor_days (float): Minimum threshold floor for LoS (in days, ~30 mins = 1/48).
            Default: 1/48.
        use_decay (bool): Whether to include exponential decay channels (0.75^j).
            Set to False for the paper's 'No Decay' ablation variant. Default: True.
        multitask (bool): Whether to predict both LoS and in-hospital mortality.
            Default: False.
        alpha (float): Weighting for mortality loss in multitask mode. Default: 0.5.

    Note on Batch Normalization over Padded Steps:
        TPC utilizes 1D Batch Normalization (BatchNorm1d) across feature channels.
        In mini-batches containing padded sequence lengths, BatchNorm computes running
        mean and variance across all time steps including padded zeros. While length-
        bucketed batching (e.g., BucketBatchSampler) minimizes padding overhead, uneven
        batches may still include zero-padded steps in the batch statistics calculation.

    Examples:
        >>> import torch
        >>> from pyhealth.models.tpc import TPC
        >>> model = TPC(dataset=None, continuous_dim=12, static_dim=5)
        >>> vitals = torch.randn(2, 24, 12)
        >>> vitals_decay = torch.rand(2, 24, 12)
        >>> static = torch.randn(2, 5)
        >>> y_true = torch.rand(2, 24, 1) + 0.1
        >>> out = model(
        ...     vitals=vitals, vitals_decay=vitals_decay, static=static, y_true=y_true
        ... )
        >>> out["y_prob"].shape
        torch.Size([2, 24, 1])
        >>> "loss" in out
        True
    """

    def __init__(
        self,
        dataset: SampleDataset | None = None,
        feature_keys: list[str] | None = None,
        label_key: str | None = None,
        mode: str | None = None,
        continuous_dim: int | None = None,
        static_dim: int | None = None,
        n_layers: int = 3,
        temp_dim: int = 8,
        point_dim: int = 16,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        loss_fn: str = "msle",
        min_time: int = 4,
        floor_days: float = 1.0 / 48.0,
        use_decay: bool = True,
        multitask: bool = False,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(dataset=cast("SampleDataset", dataset), **kwargs)
        self.label_key = label_key or (
            self.label_keys[0] if getattr(self, "label_keys", None) else "los"
        )
        self.mode = mode or "regression"

        # Infer input dimensions from dataset if provided and not explicitly given
        if dataset is not None:
            try:
                sample0 = dataset[0]
                if continuous_dim is None and "vitals" in sample0:
                    v = sample0["vitals"]
                    continuous_dim = (
                        v.shape[-1] if hasattr(v, "shape") else len(v[0])
                    )
                if static_dim is None and "static" in sample0:
                    s = sample0["static"]
                    static_dim = (
                        s.shape[-1] if hasattr(s, "shape") else len(s)
                    )
            except (AttributeError, IndexError, KeyError, TypeError):
                logger.debug(
                    "Could not automatically infer feature dimensions "
                    "from dataset."
                )

        if continuous_dim is None:
            continuous_dim = 12
        if static_dim is None:
            static_dim = 0

        self.continuous_dim = continuous_dim
        self.static_dim = static_dim
        self.n_layers = n_layers
        self.temp_dim = temp_dim
        self.point_dim = point_dim
        self.kernel_size = kernel_size
        self.loss_fn_type = loss_fn.lower()
        self.min_time = min_time
        self.floor_days = floor_days
        self.use_decay = use_decay
        self.multitask = multitask
        self.alpha = alpha

        # Stacked TPC blocks
        self.blocks = nn.ModuleList(
            [
                TPCBlock(
                    layer_idx=n,
                    num_features=continuous_dim,
                    point_dim=point_dim,
                    temp_dim=temp_dim,
                    static_dim=static_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    use_decay=use_decay,
                )
                for n in range(1, n_layers + 1)
            ]
        )

        # Final feature count after N layers: F + N * Z
        final_features = continuous_dim + n_layers * point_dim
        final_channels = temp_dim + 1
        final_in_dim = (final_features * final_channels) + static_dim

        # Prediction head: Two pointwise layers (Section 3.5 & Appendix A)
        self.head_conv1 = nn.Conv1d(final_in_dim, hidden_dim, kernel_size=1)
        self.head_bn1 = nn.BatchNorm1d(hidden_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.head_fc = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        # Optional mortality prediction head for multitask learning (Section 5.1.3)
        if self.multitask:
            self.mort_fc = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes MSLE (Eq. 6) or MSE loss averaged per stay over valid hours.

        Conforms to Equation (6) in Rocheteau et al.:
        Loss is averaged over valid time steps within each stay first, and then
        averaged across valid stays in the batch.

        Args:
            y_pred: Predicted remaining LoS tensor of shape (B, T, 1).
            y_true: Ground truth remaining LoS tensor of shape (B, T, 1).
            mask: Optional boolean or binary mask tensor of shape (B, T) or (B, T, 1).

        Returns:
            torch.Tensor: Scalar loss tensor averaged over valid stays.
        """
        # Ensure 3D: (B, T, 1)
        if y_true.dim() == 1:
            y_true = y_true.view(-1, 1, 1)
        elif y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1) if y_pred.dim() == 3 else y_true

        T = y_pred.shape[1]

        # Build valid time mask (excludes zero padding and warmup hours t < min_time)
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(-1)
        else:
            valid_mask = y_true > 0.0

        # Enforce evaluation starting at min_time (default: 4,
        # corresponding to 5h elapsed)
        if T > self.min_time:
            warmup_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            warmup_mask[:, self.min_time :, :] = True
            eval_mask = valid_mask & warmup_mask
            # If all steps for a stay fall in warmup, fallback to valid_mask
            no_eval = (eval_mask.sum(dim=(1, 2)) == 0) & (
                valid_mask.sum(dim=(1, 2)) > 0
            )
            eval_mask[no_eval] = valid_mask[no_eval]
        else:
            eval_mask = valid_mask

        # Compute per-step error
        if self.loss_fn_type == "msle":
            log_pred = torch.log(torch.clamp(y_pred, min=self.floor_days))
            log_true = torch.log(torch.clamp(y_true, min=self.floor_days))
            err = (log_pred - log_true) ** 2
        else:
            err = (y_pred - y_true) ** 2

        # Average per stay over valid hours (Eq. 6), then average across stays
        m = eval_mask.float()
        n = m.sum(dim=(1, 2))  # (B,) valid hours per patient
        per_stay = (err * m).sum(dim=(1, 2)) / n.clamp(min=1.0)
        valid_stays = n > 0
        if valid_stays.any():
            return per_stay[valid_stays].mean()
        return per_stay.mean()

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Forward propagation for TPC.

        Args:
            **kwargs: Inputs including 'vitals', optional 'vitals_decay' or
                'vitals_time_delta', optional 'static', optional 'mask', and label keys.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing 'y_prob', and optionally
                'loss', 'y_true', and 'mortality_prob'.
        """
        # 1. Extract and validate vitals
        raw_vitals = kwargs.get("vitals")
        if not isinstance(raw_vitals, torch.Tensor):
            raise TypeError(
                f"Expected 'vitals' to be a torch.Tensor, but got "
                f"{type(raw_vitals).__name__}"
            )

        # Ensure vitals has shape (batch_size, seq_len, continuous_dim)
        vitals_tensor = (
            raw_vitals if raw_vitals.dim() == 3 else raw_vitals.unsqueeze(1)
        )
        B, T, _ = vitals_tensor.shape

        # 2. Extract or compute decay indicators: 0.75^j (if use_decay is enabled)
        x_raw = vitals_tensor.transpose(1, 2).float()
        if self.use_decay:
            raw_decay = kwargs.get("vitals_decay")
            if isinstance(raw_decay, torch.Tensor):
                vitals_decay = raw_decay
            else:
                raw_time_delta = kwargs.get("vitals_time_delta")
                if isinstance(raw_time_delta, torch.Tensor):
                    vitals_decay = torch.pow(
                        0.75, torch.clamp(raw_time_delta, min=0.0)
                    )
                else:
                    vitals_decay = torch.ones_like(vitals_tensor)
            x_decay = vitals_decay.transpose(1, 2).float()
            # Construct Layer 1 Input: 2 channels per feature [x'_t, x''_t]
            h = torch.stack([x_raw, x_decay], dim=2)  # (B, F, 2, T)
        else:
            x_decay = None
            # Ablation variant without decay indicators: 1 channel per feature [x'_t]
            h = x_raw.unsqueeze(2)  # (B, F, 1, T)

        # 3. Extract static features (if any)
        raw_static = kwargs.get("static")
        if raw_static is None:
            raw_static = kwargs.get("static_demographics")

        s_expanded: torch.Tensor | None = None
        if self.static_dim > 0:
            if isinstance(raw_static, torch.Tensor):
                s_tensor = raw_static.float()
                if s_tensor.dim() == 1:
                    s_tensor = s_tensor.unsqueeze(0)
                if s_tensor.shape[1] < self.static_dim:
                    s_tensor = F.pad(
                        s_tensor, (0, self.static_dim - s_tensor.shape[1])
                    )
                elif s_tensor.shape[1] > self.static_dim:
                    s_tensor = s_tensor[:, : self.static_dim]
                s_expanded = s_tensor.unsqueeze(-1).expand(-1, -1, T)
            else:
                s_expanded = torch.zeros(
                    B, self.static_dim, T, device=vitals_tensor.device
                )

        # 4. Pass through stacked TPC blocks
        prev_pointwise: list[torch.Tensor] = []
        for block in self.blocks:
            h, point_out_ch = block(
                h, x_raw, x_decay, s_expanded, prev_pointwise
            )
            prev_pointwise.append(point_out_ch)

        # 5. Prediction Head (Section 3.5 & Appendix A)
        final_R = self.continuous_dim + self.n_layers * self.point_dim
        final_C = self.temp_dim + 1
        h_final = h.view(B, final_R * final_C, T)

        if s_expanded is not None and self.static_dim > 0:
            final_in = torch.cat([h_final, s_expanded], dim=1)
        else:
            final_in = h_final

        # Pointwise conv -> BatchNorm -> ReLU -> Dropout (Appendix A)
        feat = self.head_dropout(
            F.relu(self.head_bn1(self.head_conv1(final_in)))
        )
        los_out = self.head_fc(feat).transpose(1, 2)  # (B, T, 1)

        # Softplus with matching floor prevents dying ReLUs and log singularities
        y_prob = F.softplus(los_out) + self.floor_days  # (B, T, 1)

        raw_mask = kwargs.get("mask")
        mask_tensor = raw_mask if isinstance(raw_mask, torch.Tensor) else None

        result: dict[str, torch.Tensor] = {
            "y_prob": y_prob,
            "logit": los_out,
        }

        # 6. Loss Calculation
        raw_y_true = kwargs.get(self.label_key)
        if raw_y_true is None:
            raw_y_true = kwargs.get("y_true")

        if isinstance(raw_y_true, torch.Tensor):
            y_true_tensor = raw_y_true.float()
            result["y_true"] = y_true_tensor
            result["loss"] = self._compute_loss(
                y_prob, y_true_tensor, mask=mask_tensor
            )

            # Multitask in-hospital mortality loss using numerically stable logits
            if self.multitask and hasattr(self, "mort_fc"):
                mort_out = self.mort_fc(feat).transpose(1, 2)  # (B, T, 1) logits
                result["mortality_prob"] = torch.sigmoid(mort_out)
                result["mortality_logit"] = mort_out

                raw_mort = kwargs.get("mortality")
                if raw_mort is None:
                    raw_mort = kwargs.get("in_hospital_mortality")
                if isinstance(raw_mort, torch.Tensor):
                    mort_true = (
                        raw_mort.float().view(-1, 1, 1).expand_as(mort_out)
                    )
                    if mask_tensor is not None:
                        m_valid = mask_tensor.bool().unsqueeze(-1)
                        # Apply matching warmup mask (t >= min_time)
                        if T > self.min_time:
                            warmup_m = torch.zeros_like(m_valid)
                            warmup_m[:, self.min_time :, :] = True
                            m_eval = m_valid & warmup_m
                            no_eval = (m_eval.sum(dim=(1, 2)) == 0) & (
                                m_valid.sum(dim=(1, 2)) > 0
                            )
                            m_eval[no_eval] = m_valid[no_eval]
                        else:
                            m_eval = m_valid
                        if m_eval.sum() > 0:
                            mort_loss = F.binary_cross_entropy_with_logits(
                                mort_out[m_eval], mort_true[m_eval]
                            )
                        else:
                            mort_loss = F.binary_cross_entropy_with_logits(
                                mort_out, mort_true
                            )
                    else:
                        mort_loss = F.binary_cross_entropy_with_logits(
                            mort_out, mort_true
                        )
                    result["loss"] = result["loss"] + self.alpha * mort_loss
        elif self.multitask and hasattr(self, "mort_fc"):
            mort_out = self.mort_fc(feat).transpose(1, 2)
            result["mortality_prob"] = torch.sigmoid(mort_out)
            result["mortality_logit"] = mort_out

        return result