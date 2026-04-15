import json
import os
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
from torch.utils.data import DataLoader

from tqdm import tqdm
from app.core.config import MacroConfig

class Processed():
    def __init__(self, config:MacroConfig, base_dir, processed_check=False, **kwargs):
        self.base_dir = base_dir
        self.config = config
        self.input_size = 0
        self.processed_check = processed_check
        self.max_len = 100  

    def generate_indicators(self, path):
        data_path = path

        with open(data_path, "r", encoding="utf-8") as f:
            data_list:list[dict] = json.load(f)

        T = []
        current_seq = []

        for data in tqdm(data_list, desc="Processing"):
            if data.get('deltatime') == 0:
                continue            
            
            x_diff = data.get('cur_x') - data.get('pre_x')
            y_diff = data.get('cur_y') - data.get('pre_y')

           
            if x_diff == 0 and y_diff == 0 and data.get('status') != "END":
                continue
            
            point = [
                int( (x_diff / max(data.get('pre_x'), 1)) * 100) / 100,
                int( (y_diff / max(data.get('pre_x'), 1)) * 100) / 100,
                data.get('deltatime') * 100
            ]
            
            if data["status"] != "END":
                current_seq.append(point)
            
            if data["status"] == "END":
                T.append(current_seq)
                current_seq = []

        if self.processed_check:
            self.show_plot(T)
            
        padded_T = []
        for seq in T:
            pad_size = self.max_len - len(seq)
            if pad_size > 0:
                seq.extend([[-1, -1, -1]] * pad_size)
            padded_T.append(seq)
        
        if self.processed_check:
            self.show_pad(padded_T)

        numpy_T = np.array(T)
        print(f" T Shape : {numpy_T.shape}")

        self.input_size = 3
        return numpy_T
        
    def show_pad(self, padded_T):
        padded_array = np.array(padded_T)  # (N, max_len, 3)
        is_pad = np.all(padded_array == -1, axis=-1)  # (N, max_len)
        seq_lengths = (~is_pad).sum(axis=1)
        pad_ratio = is_pad.sum() / is_pad.size

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("PAD Distribution", fontsize=14)

        axes[0].hist(seq_lengths, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
        axes[0].axvline(seq_lengths.mean(), color="red", linestyle="--", linewidth=1, label=f"mean={seq_lengths.mean():.1f}")
        axes[0].set_title("Sequence length distribution")
        axes[0].set_xlabel("actual length (non-PAD steps)")
        axes[0].set_ylabel("count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        pad_per_step = is_pad.mean(axis=0)
        axes[1].plot(pad_per_step, color="coral", linewidth=1.2)
        axes[1].fill_between(range(len(pad_per_step)), pad_per_step, alpha=0.2, color="coral")
        axes[1].set_title("PAD ratio per position")
        axes[1].set_xlabel("step position")
        axes[1].set_ylabel("PAD ratio")
        axes[1].grid(True, alpha=0.3)

        sample_n = min(200, len(padded_T))
        axes[2].imshow(is_pad[:sample_n].astype(int), aspect="auto", cmap="Blues", interpolation="nearest")
        axes[2].set_title(f"PAD heatmap (first {sample_n} seqs)")
        axes[2].set_xlabel("step position")
        axes[2].set_ylabel("sequence index")
        axes[2].text(0.98, 0.02, f"overall PAD {pad_ratio*100:.1f}%",
                    transform=axes[2].transAxes, ha="right", va="bottom",
                    fontsize=9, color="gray")

        plt.tight_layout()
        plt.show()

    def show_plot(self, T):
        seq = T[0]  # shape: (N, 3)

        x_vals  = [p[0] for p in seq if p[0] != -1]
        y_vals  = [p[1] for p in seq if p[1] != -1]
        dt_vals = [p[2] for p in seq if p[2] != -1]
        steps   = list(range(len(x_vals)))

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("T[0] Sequence", fontsize=14)

        axes[0].plot(steps, x_vals, color="steelblue", linewidth=1.2)
        axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[0].set_ylabel("x_diff (norm)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, y_vals, color="coral", linewidth=1.2)
        axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1].set_ylabel("y_diff (norm)")
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(steps, dt_vals, color="mediumseagreen", width=0.8, alpha=0.8)
        axes[2].set_ylabel("deltatime × 100")
        axes[2].set_xlabel("step")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    
    def generation_procceed_data(self, numpy_array, inference_mode=False) -> DataLoader:
        ratio = self.config.train_val_ratio
        if inference_mode:
            ratio = 1

        train_size = int(ratio * len(numpy_array))
        
        train_dataset = numpy_array[:train_size]
        val_dataset = numpy_array[train_size:]

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
        )

        return train_loader, val_loader
