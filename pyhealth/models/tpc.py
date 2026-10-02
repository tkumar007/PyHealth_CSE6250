import torch
import torch.nn as nn
from pyhealth.models import BaseModel

class TPCLayer(nn.Module):
    """A single Temporal Pointwise Convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TPCLayer, self).__init__()
        
        # Setting groups=in_channels strictly forces the layer to convolve 
        # each clinical feature completely independently across time.
        self.temporal_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding="same" 
        )
        
        # A 1x1 convolution that looks at a single time step and mixes the 
        # independent features together into the output channels.
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.pointwise_conv(x)
        return self.activation(x)


class TPC(BaseModel):
    """
    Temporal Pointwise Convolution model for Length of Stay prediction.
    Implements the exponential decay mechanism for missing EHR data.
    """
    def __init__(
        self, 
        dataset,
        feature_keys=None, 
        label_key=None,   
        mode=None,        
        continuous_dim=12,  
        hidden_dim=64, 
        kernel_size=3,
        **kwargs
    ):
        super(TPC, self).__init__(dataset=dataset, **kwargs)
        self.label_key = label_key or (self.label_keys[0] if getattr(self, "label_keys", None) else "los")
        
        self.tpc_block = TPCLayer(
            in_channels=continuous_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size
        )
        
        # Missing Data Decay Parameter
        self.decay_weight = nn.Parameter(torch.Tensor(continuous_dim))
        nn.init.uniform_(self.decay_weight, 0.0, 0.1)
        
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, **kwargs):
        vitals = kwargs.get("vitals")
        vitals_time_delta = kwargs.get("vitals_time_delta")
        y_true = kwargs.get(self.label_key, kwargs.get("y_true"))
        
        assert isinstance(vitals, torch.Tensor), "vitals is missing or not a Tensor"

        if vitals_time_delta is None:
            vitals_time_delta = torch.zeros_like(vitals)
        else:
            assert isinstance(vitals_time_delta, torch.Tensor), "vitals_time_delta missing/invalid"
        
        # Apply missing data exponential decay: exp(-W * delta_t)
        positive_decay = torch.relu(self.decay_weight)
        decay_factor = torch.exp(-positive_decay * vitals_time_delta)
        decayed_vitals = vitals * decay_factor
        
        x = decayed_vitals.transpose(1, 2)
        
        x = self.tpc_block(x)
        
        x = torch.max(x, dim=2)[0]
        
        y_prob = self.fc(x)
        
        loss = None
        if y_true is not None:
            assert isinstance(y_true, torch.Tensor)
            loss_fn = nn.MSELoss()
            y_true = y_true.view(-1, 1).float()
            loss = loss_fn(y_prob, y_true)
            
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true
        }