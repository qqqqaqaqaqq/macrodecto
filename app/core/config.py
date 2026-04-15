import torch
from dataclasses import dataclass

# 16 ~ 32 => lr 5 * 10^-5
# 48 ~ 128 => lr 1 * 10^-4
# 256 ~ 512 => lr 2 * 10^-4

@dataclass
class MacroConfig():

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size: int = 256
    epochs: int = 100
    lr: float = 2e-4
    accum_steps: int = 1
    weight_decay: float = 0.01

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    rope_hz:int = 100

    train_val_ratio: float = 0.8

    use_amp: bool = False
    dtype:torch.types = torch.bfloat16

# 실행
macro_config = MacroConfig()