import os
import torch
from dataclasses import asdict

import torch.nn.functional as F
from app.models.MacroDetector import MacroDetector
from app.core.config import MacroConfig

class EncoderInference():
    def __init__(self, config:MacroConfig, base_dir, input_size:int, **kwargs):
        self.base_dir = base_dir
        self.config = config
        self.device = self.config.device
        self.model_dir = os.path.join(base_dir, "weights", "encoder")

        param = asdict(config)
        print(f"config : {param}")

        self.model = MacroDetector(input_size=input_size, **param)

        print(self.model_dir)

        model_path = os.path.join(self.model_dir, "encoder_macro_model.pth")
        self.checkpoint = None
        if not os.path.exists(model_path):
            print("model 확인 불가")
            return None
        
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        print(f"check_point_loss : {self.checkpoint['val_loss']}")
        self.model = self.model.to(self.device, dtype=torch.float32)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"총 파라미터 수: {total_params:,}")

    @torch.no_grad()
    def validation(self, loader):
        self.model.eval()
        val_loss = 0

        # 정상 범주 에러율 측정
        for batch_x in loader:
            batch_x = batch_x.to(self.device, dtype=self.config.dtype, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=self.config.dtype, enabled=self.config.use_amp):
                prediction, rand_mask = self.model(src=batch_x) 
                inverted_mask = 1 - rand_mask 

                masked_prediction = prediction * inverted_mask
                masked_target = batch_x * inverted_mask
                recon_loss = F.smooth_l1_loss(masked_prediction, masked_target, beta=0.5, reduction='sum')
                recon_loss = recon_loss / (inverted_mask.sum() * batch_x.size(-1) + 1e-9)

                total_loss = recon_loss

            val_loss += total_loss.item() * batch_x.size(0)        
        
        return val_loss / len(loader.dataset)

    def run(self, loader):
        return self.validation(loader=loader)