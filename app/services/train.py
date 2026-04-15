import os
import torch
from tqdm import tqdm

import torch.nn.functional as F
from dataclasses import asdict
from app.core.config import MacroConfig
from app.models.MacroDetector import MacroDetector
from torch.utils.tensorboard import SummaryWriter

class EncoderTrain():
    def __init__(self, config:MacroConfig, base_dir, input_size:int, **kwargs):
        self.base_dir = base_dir
        self.config = config
        self.device = self.config.device
        self.model_dir = os.path.join(base_dir, "weights", "encoder")
        os.makedirs(self.model_dir, exist_ok=True)

        self.save_path = os.path.join(self.model_dir, "epochs")
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "logs"))

        param = asdict(config)

        print(f"config : {param}")

        self.model = MacroDetector(
            input_size=input_size,
            **param
        )

        model_path = os.path.join(self.model_dir, "encoder_macro_model.pth")
        self.checkpoint = None
        if os.path.exists(model_path):
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(self.checkpoint["model_state_dict"])

            print(f"check_point_loss : {self.checkpoint['val_loss']}")

        self.model = self.model.to(self.device, dtype=torch.float32)

    def train(self, train_loader, optimizer, grad_check):
        self.model.train()
        total_train_loss = 0
        
        for batch_x in train_loader:
            batch_x = batch_x.to(self.device, dtype=self.config.dtype, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=self.config.dtype, enabled=self.config.use_amp):
                prediction, rand_mask = self.model(src=batch_x) 
                inverted_mask = 1 - rand_mask 

                masked_prediction = prediction * inverted_mask
                masked_target = batch_x * inverted_mask
                recon_loss = F.smooth_l1_loss(masked_prediction, masked_target, beta=0.5, reduction='sum')
                recon_loss = recon_loss / (inverted_mask.sum() * batch_x.size(-1) + 1e-9)

                total_loss = recon_loss

                total_loss.backward()

            if not grad_check:
                self.grad_check(self.model)
                grad_check = True

            optimizer.step()
            total_train_loss += total_loss.item() * batch_x.size(0)

        return total_train_loss / len(train_loader.dataset), grad_check
    
    @torch.no_grad()
    def validation(self, val_loader):
        self.model.eval()
        total_val_loss = 0

        for batch_x in val_loader:
            batch_x = batch_x.to(self.device, dtype=self.config.dtype, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=self.config.dtype, enabled=self.config.use_amp):
                prediction, rand_mask = self.model(src=batch_x) 
                inverted_mask = 1 - rand_mask 

                masked_prediction = prediction * inverted_mask
                masked_target = batch_x * inverted_mask
                recon_loss = F.smooth_l1_loss(masked_prediction, masked_target, beta=0.5, reduction='sum')
                recon_loss = recon_loss / (inverted_mask.sum() * batch_x.size(-1) + 1e-9)

                total_loss = recon_loss

            total_val_loss += total_loss.item() * batch_x.size(0)

        return total_val_loss / len(val_loader.dataset)

    def grad_check(self, model):
        print("\n[Gradient Check]")
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad
                print(f"  {name:<50} | mean={g.mean().item():+.6f} | std={g.std().item():.6f} | max={g.abs().max().item():.6f}")
            else:
                print(f"  {name:<50} | grad 없음")

    def run(self, train_loader, val_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
        )

        start_epoch = 0
        if self.checkpoint:
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in self.checkpoint:
                scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
            start_epoch = self.checkpoint['epoch'] + 1

        os.makedirs(self.save_path, exist_ok=True)

        best_val_loss = float('inf')

        grad_check = False

        epoch_pbar = tqdm(range(start_epoch, self.config.epochs), desc="Training Progress")
        for epoch in epoch_pbar:
            avg_train_loss, return_grad_check = self.train(train_loader=train_loader, optimizer=optimizer, grad_check=grad_check)
            grad_check = return_grad_check
            avg_val_loss = self.validation(val_loader=val_loader)

            scheduler.step(avg_val_loss)
        
            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                        
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }

            if epoch % 30 == 0:
                torch.save(checkpoint, os.path.join(self.save_path, f"model_ep{epoch}_{avg_val_loss:.4f}.pth"))

            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, os.path.join(self.model_dir, "encoder_macro_model.pth"))
