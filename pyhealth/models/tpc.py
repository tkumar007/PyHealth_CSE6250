import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel


class TPCBlock(nn.Module):
    """A single Temporal Pointwise Convolution (TPC) block.
    
    Implements Equations (1)-(5) and Figure 3 from:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'
    (Rocheteau et al., ACM CHIL 2021).
    
    Args:
        layer_idx (int): 1-indexed layer number (n).
        num_features (int): Number of initial time-series features (F).
        point_dim (int): Pointwise output channels (Z).
        temp_dim (int): Temporal output channels per feature (Y).
        static_dim (int): Number of static features (S).
        kernel_size (int): Filter size (k).
        dropout (float): Dropout probability.
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
    ):
        super(TPCBlock, self).__init__()
        self.n = layer_idx
        self.k = kernel_size
        self.d = layer_idx  # Dilation increments by 1 with each layer (d = n)
        self.pad = self.d * (self.k - 1)  # Causal left-sided padding
        
        self.R_n = num_features + point_dim * (self.n - 1)  # Number of features
        self.C_n = 2 if self.n == 1 else (temp_dim + 1)     # Channels per feature
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
        # Pointwise input dimension: flattened temporal input + static features + decay indicators
        self.P_n = (self.R_n * self.C_n) + self.static_dim + num_features
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
        x_decay: torch.Tensor,
        s_expanded: torch.Tensor | None,
        prev_pointwise: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of a single TPC block.
        
        Args:
            h: Input tensor of shape (batch, R_n, C_n, seq_len).
            x_raw: Raw time series of shape (batch, F, seq_len).
            x_decay: Decay indicators of shape (batch, F, seq_len).
            s_expanded: Static features of shape (batch, S, seq_len) or None.
            prev_pointwise: List of pointwise outputs from previous layers.
            
        Returns:
            h_next: Output tensor of shape (batch, R_{n+1}, Y+1, seq_len).
            point_out_channel: Pointwise output of shape (batch, Z, 1, seq_len).
        """
        B, _, _, T = h.shape
        
        # --- A. Temporal Convolution Branch ---
        h_flat = h.view(B, self.R_n * self.C_n, T)
        # Left-sided causal padding ensures no information leakage from the future
        h_pad = F.pad(h_flat, (self.pad, 0))
        temp_out = self.temporal_conv(h_pad)
        temp_out = self.temp_dropout(self.temp_bn(temp_out))
        temp_out = temp_out.view(B, self.R_n, self.temp_dim, T)
        
        # --- B. Pointwise Convolution Branch ---
        point_in_list = [h_flat]
        if s_expanded is not None and self.static_dim > 0:
            point_in_list.append(s_expanded)
        point_in_list.append(x_decay)
        point_in = torch.cat(point_in_list, dim=1)
        
        point_out = self.pointwise_conv(point_in)
        point_out = self.point_dropout(self.point_bn(point_out))  # (B, Z, T)
        
        # --- C. Skip Connections (Eq. 4) ---
        if self.n == 1:
            r_n = torch.cat([temp_out, x_raw.unsqueeze(2)], dim=2)  # (B, F, Y+1, T)
        else:
            skips = torch.cat([x_raw.unsqueeze(2)] + prev_pointwise, dim=1)
            r_n = torch.cat([temp_out, skips], dim=2)  # (B, R_n, Y+1, T)
            
        # --- D. Parallel Feature Concatenation & Activation (Eq. 5) ---
        # Broadcast pointwise output across the channel dimension (Y + 1 times)
        point_broadcast = point_out.unsqueeze(2).repeat(1, 1, self.temp_dim + 1, 1)
        h_next = F.relu(torch.cat([r_n, point_broadcast], dim=1))  # (B, R_n + Z, Y+1, T)
        
        return h_next, point_out.unsqueeze(2)


class TPC(BaseModel):
    """Temporal Pointwise Convolution (TPC) model for ICU Length of Stay prediction.
    
    Faithfully reproduces the architecture from:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'
    (Emma Rocheteau, Pietro Liò, & Stephanie Hyland, ACM CHIL 2021).
    
    Args:
        dataset: The PyHealth dataset.
        feature_keys: List of input feature keys (e.g. ['vitals', 'static']).
        label_key: Target key (defaults to 'los').
        mode: Prediction mode ('regression').
        continuous_dim (int): Number of time series features (F, e.g. 12, 87, or 101). Default: 12.
        static_dim (int): Number of static features (S). Default: 0.
        n_layers (int): Number of stacked TPC blocks (N). Default: 3.
        temp_dim (int): Temporal output channels per feature (Y). Default: 8.
        point_dim (int): Pointwise output channels (Z). Default: 16.
        kernel_size (int): Temporal filter size (k). Default: 3.
        hidden_dim (int): Hidden dimension for prediction head. Default: 64.
        dropout (float): Dropout probability. Default: 0.1.
        loss_fn (str): Loss function to use ('msle' or 'mse'). Default: 'msle'.
        min_time (int): Prediction hour from which loss is computed. Default: 5.
        sequence_output (bool): If True, returns hourly predictions (B, T, 1). If False,
            pools over time for a single patient summary prediction (B, 1). Default: True.
        multitask (bool): Whether to predict both LoS and in-hospital mortality. Default: False.
        alpha (float): Weighting for mortality loss in multitask mode. Default: 0.5.
    """
    def __init__(
        self,
        dataset,
        feature_keys: list[str] | None = None,
        label_key: str | None = None,
        mode: str | None = None,
        continuous_dim: int = 12,
        static_dim: int = 0,
        n_layers: int = 3,
        temp_dim: int = 8,
        point_dim: int = 16,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        loss_fn: str = "msle",
        min_time: int = 5,
        sequence_output: bool = True,
        multitask: bool = False,
        alpha: float = 0.5,
        **kwargs,
    ):
        super(TPC, self).__init__(dataset=dataset, **kwargs)
        self.label_key = label_key or (self.label_keys[0] if getattr(self, "label_keys", None) else "los")
        self.mode = mode or "regression"
        
        self.continuous_dim = continuous_dim
        self.static_dim = static_dim
        self.n_layers = n_layers
        self.temp_dim = temp_dim
        self.point_dim = point_dim
        self.kernel_size = kernel_size
        self.loss_fn_type = loss_fn.lower()
        self.min_time = min_time
        self.sequence_output = sequence_output
        self.multitask = multitask
        self.alpha = alpha
        
        # Stacked TPC blocks
        self.blocks = nn.ModuleList([
            TPCBlock(
                layer_idx=n,
                num_features=continuous_dim,
                point_dim=point_dim,
                temp_dim=temp_dim,
                static_dim=static_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for n in range(1, n_layers + 1)
        ])
        
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
        """Computes MSLE (Eq. 6) or MSE loss averaged per stay over valid hours (t >= min_time)."""
        # Ensure 3D: (B, T, 1)
        if y_true.dim() == 1:
            y_true = y_true.view(-1, 1, 1)
        elif y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1) if y_pred.dim() == 3 else y_true
            
        B, T = y_pred.shape[0], y_pred.shape[1]
        
        # Build valid time mask (excludes zero padding and warmup hours t < min_time)
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(-1)
        else:
            valid_mask = (y_true > 0.0)
            
        # Enforce evaluation starting at min_time (default: 5 hours, Section 5.1.1)
        if T > self.min_time:
            warmup_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            warmup_mask[:, self.min_time:, :] = True
            combined_mask = valid_mask & warmup_mask
            # If stay is shorter than min_time or all warmup masked, fallback to valid_mask for that patient
            for b in range(B):
                if combined_mask[b].sum() == 0 and valid_mask[b].sum() > 0:
                    combined_mask[b] = valid_mask[b]
            eval_mask = combined_mask
        else:
            eval_mask = valid_mask

        # Compute loss per patient (Eq. 6 averages over T per stay first)
        patient_losses = []
        for b in range(B):
            m_b = eval_mask[b].squeeze(-1)
            if m_b.sum() == 0:
                continue
            y_p_b = y_pred[b, m_b]
            y_t_b = y_true[b, m_b]
            
            if self.loss_fn_type == "msle":
                log_pred = torch.log(torch.clamp(y_p_b, min=1e-5))
                log_true = torch.log(torch.clamp(y_t_b, min=1e-5))
                patient_losses.append(torch.mean((log_pred - log_true) ** 2))
            else:
                patient_losses.append(F.mse_loss(y_p_b, y_t_b))
                
        if len(patient_losses) > 0:
            return torch.stack(patient_losses).mean()
            
        # Fallback if no valid mask entries
        if self.loss_fn_type == "msle":
            return torch.mean((torch.log(torch.clamp(y_pred, min=1e-5)) - torch.log(torch.clamp(y_true, min=1e-5))) ** 2)
        return F.mse_loss(y_pred, y_true)

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Forward propagation for TPC.
        
        Args:
            **kwargs: Inputs including 'vitals', optional 'vitals_decay' or
                'vitals_time_delta', optional 'static', optional 'mask', and label keys.
                
        Returns:
            Dictionary containing 'y_prob', and optionally 'loss', 'y_true',
            and 'mortality_prob'.
        """
        # 1. Extract and validate vitals
        raw_vitals = kwargs.get("vitals")
        assert isinstance(raw_vitals, torch.Tensor), "vitals is missing or not a Tensor"
        
        # Ensure vitals has shape (batch_size, seq_len, continuous_dim)
        vitals_tensor = raw_vitals if raw_vitals.dim() == 3 else raw_vitals.unsqueeze(1)
        B, T, F_dim = vitals_tensor.shape
        
        # 2. Extract or compute decay indicators: 0.75^j
        raw_decay = kwargs.get("vitals_decay")
        if isinstance(raw_decay, torch.Tensor):
            vitals_decay = raw_decay
        else:
            raw_time_delta = kwargs.get("vitals_time_delta")
            if isinstance(raw_time_delta, torch.Tensor):
                vitals_decay = torch.pow(0.75, torch.clamp(raw_time_delta, min=0.0))
            else:
                vitals_decay = torch.ones_like(vitals_tensor)
        
        # Transpose for PyTorch Conv1d: (B, F, T)
        x_raw = vitals_tensor.transpose(1, 2).float()
        x_decay = vitals_decay.transpose(1, 2).float()
        
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
                    s_tensor = F.pad(s_tensor, (0, self.static_dim - s_tensor.shape[1]))
                elif s_tensor.shape[1] > self.static_dim:
                    s_tensor = s_tensor[:, :self.static_dim]
                s_expanded = s_tensor.unsqueeze(-1).expand(-1, -1, T)
            else:
                s_expanded = torch.zeros(B, self.static_dim, T, device=vitals_tensor.device)
                
        # 4. Construct Layer 1 Input: 2 channels per feature [x'_t, x''_t] (Footnote 1)
        h = torch.stack([x_raw, x_decay], dim=2)  # (B, F, 2, T)
        
        # 5. Pass through stacked TPC blocks
        prev_pointwise: list[torch.Tensor] = []
        for block in self.blocks:
            h, point_out_ch = block(h, x_raw, x_decay, s_expanded, prev_pointwise)
            prev_pointwise.append(point_out_ch)
            
        # 6. Prediction Head (Section 3.5 & Appendix A)
        final_R = self.continuous_dim + self.n_layers * self.point_dim
        final_C = self.temp_dim + 1
        h_final = h.view(B, final_R * final_C, T)
        
        if s_expanded is not None and self.static_dim > 0:
            final_in = torch.cat([h_final, s_expanded], dim=1)
        else:
            final_in = h_final
            
        feat = self.head_dropout(self.head_bn1(F.relu(self.head_conv1(final_in))))
        los_out = self.head_fc(feat).transpose(1, 2)  # (B, T, 1)
        
        # Softplus guarantees smooth positive gradients and prevents dying ReLUs
        y_prob = F.softplus(los_out) + 1e-3
        
        raw_mask = kwargs.get("mask")
        mask_tensor = raw_mask if isinstance(raw_mask, torch.Tensor) else None
        
        if not self.sequence_output:
            # Pick the true last valid step of each patient (avoiding zero-padded steps)
            if mask_tensor is not None:
                lengths = mask_tensor.sum(dim=1).long()
                last_indices = torch.clamp(lengths - 1, min=0)
                y_prob = torch.stack([y_prob[b, last_indices[b], :] for b in range(B)], dim=0)
            else:
                y_prob = y_prob[:, -1, :]  # (B, 1)
            
        result: dict[str, torch.Tensor] = {
            "y_prob": y_prob,
        }
        
        # 7. Loss Calculation
        raw_y_true = kwargs.get(self.label_key)
        if raw_y_true is None:
            raw_y_true = kwargs.get("y_true")
            
        if isinstance(raw_y_true, torch.Tensor):
            y_true_tensor = raw_y_true.float()
            result["y_true"] = y_true_tensor
            result["loss"] = self._compute_loss(y_prob, y_true_tensor, mask=mask_tensor)
            
            # Multitask in-hospital mortality loss (evaluated across valid timesteps)
            if self.multitask and hasattr(self, "mort_fc"):
                mort_out = self.mort_fc(feat).transpose(1, 2)
                mort_prob = torch.sigmoid(mort_out)
                result["mortality_prob"] = mort_prob
                
                raw_mort = kwargs.get("mortality")
                if raw_mort is None:
                    raw_mort = kwargs.get("in_hospital_mortality")
                if isinstance(raw_mort, torch.Tensor):
                    mort_true = raw_mort.float().view(-1, 1, 1).expand_as(mort_prob)
                    if mask_tensor is not None:
                        m_valid = mask_tensor.bool().unsqueeze(-1)
                        if m_valid.sum() > 0:
                            mort_loss = F.binary_cross_entropy(mort_prob[m_valid], mort_true[m_valid])
                        else:
                            mort_loss = F.binary_cross_entropy(mort_prob, mort_true)
                    else:
                        mort_loss = F.binary_cross_entropy(mort_prob, mort_true)
                    result["loss"] = result["loss"] + self.alpha * mort_loss
        elif self.multitask and hasattr(self, "mort_fc"):
            mort_out = self.mort_fc(feat).transpose(1, 2)
            result["mortality_prob"] = torch.sigmoid(mort_out)
            
        return result