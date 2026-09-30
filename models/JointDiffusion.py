"""Sequence-preserving masked x-prediction head for FSQ-MARDM.

This module keeps the FSQ token sequence in [B, L, D] form throughout
training and sampling. Padding is visible to the attention mask, while
masked tokens can attend to each other.
"""
from typing import Optional
import torch
import torch.nn as nn
from models.DiffTransformer import DiffTransformer


class JointSequenceXPred(nn.Module):
    def __init__(self, target_channels, z_channels, hidden_size=512, depth=8,
                 num_heads=8, max_seq_len=64, num_sampling_steps=12,
                 status_scale=0.25):
        super().__init__()
        self.in_channels = target_channels
        self.num_sampling_steps = int(num_sampling_steps)
        self.status_scale = float(status_scale)
        self.status_proj = nn.Linear(1, z_channels)
        self.net = DiffTransformer(
            target_channels=target_channels,
            z_channels=z_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=0.1,
        )

    def _context(self, z, status):
        if status is None:
            return z
        return z + self.status_scale * self.status_proj(status.float().unsqueeze(-1))

    @staticmethod
    def _valid_mask(padding_mask, shape, device):
        if padding_mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        return ~padding_mask.bool()

    def forward(self, target, z, padding_mask=None, mask=None):
        """Masked sequence x-prediction loss, balanced per sequence."""
        if target.dim() != 3 or z.dim() != 3:
            raise ValueError("JointSequenceXPred expects target,z as [B,L,*]")
        b, l, _ = target.shape
        valid = self._valid_mask(padding_mask, (b, l), target.device)
        update = valid if mask is None else (mask.bool() & valid)
        t = torch.rand(b, device=target.device) * 0.98 + 0.01
        eps = torch.randn_like(target)
        te = t[:, None, None]
        noisy = torch.where(update[..., None], te * target + (1.0 - te) * eps, target)
        noisy = torch.where(valid[..., None], noisy, torch.zeros_like(noisy))
        status = update.to(target.dtype)
        pred = self.net(noisy, t, self._context(z, status), padding_mask)
        err = (pred - target).pow(2).mean(dim=-1)
        denom = update.sum(dim=-1).clamp_min(1).to(err.dtype)
        return ((err * update).sum(dim=-1) / denom).mean()

    @torch.no_grad()
    def sample(self, z, padding_mask=None, mask=None, current=None,
               temperature=1.0, cfg=1.0):
        """Generate [B,L,D], preserving visible tokens and zeroing padding."""
        if z.dim() != 3:
            raise ValueError("JointSequenceXPred.sample expects z as [B,L,H]")
        if cfg != 1.0:
            if z.shape[0] % 2:
                raise ValueError("CFG context batch must be even")
            b = z.shape[0] // 2
            zc, zu = z[:b], z[b:]
            pm = None if padding_mask is None else padding_mask[:b]
            mm = None if mask is None else mask[:b]
        else:
            b, zc, zu = z.shape[0], z, None
            pm, mm = padding_mask, mask
        l = zc.shape[1]
        valid = self._valid_mask(pm, (b, l), zc.device)
        update = valid if mm is None else (mm.bool() & valid)
        if current is None:
            current = torch.zeros(b, l, self.in_channels, device=zc.device, dtype=zc.dtype)
        else:
            current = current.to(device=zc.device, dtype=zc.dtype)
        current = torch.where(valid[..., None], current, torch.zeros_like(current))
        state = torch.where(update[..., None],
                            torch.randn_like(current) * float(temperature), current)
        state = torch.where(valid[..., None], state, torch.zeros_like(state))
        steps = max(1, self.num_sampling_steps)
        ts = torch.linspace(0.02, 0.98, steps, device=zc.device)
        for i, tval in enumerate(ts):
            t = torch.full((b,), float(tval), device=zc.device, dtype=zc.dtype)
            status = update.to(zc.dtype)
            pred_c = self.net(state, t, self._context(zc, status), pm)
            if zu is not None:
                pred_u = self.net(state, t, self._context(zu, status), pm)
                pred = pred_u + float(cfg) * (pred_c - pred_u)
            else:
                pred = pred_c
            pred = pred.clamp(-1.25, 1.25)
            if i + 1 < steps:
                tn = ts[i + 1].to(dtype=zc.dtype)
                vel = (pred - state) / (1.0 - t[:, None, None]).clamp_min(1e-3)
                nxt = state + (tn - t)[:, None, None] * vel
            else:
                nxt = pred
            state = torch.where(update[..., None], nxt, current)
            state = torch.where(valid[..., None], state, torch.zeros_like(state))
        return torch.where(update[..., None], state, current)

