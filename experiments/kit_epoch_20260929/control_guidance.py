"""Training-free guidance experiments for the unchanged FSQ + JiT generator.

Borrowed mechanisms: Adaptive Projected Guidance (APG), guidance confined to a
sampling interval, and CFG rescaling. These are adaptations to six-dimensional
clean-coordinate predictions, not new network modules or claims of invention.
All CONFIGS values are preregistered project trials, NOT established optima for
motion. APG norm clipping is disabled (threshold=0), because an image-space
absolute radius is not justified in the six-dimensional FSQ coordinate space.

The original outer CFG remains 4.5 and the original two head calls per inner
step are retained, including when interval guidance is inactive. Projection,
momentum, norms, and standard deviations operate only over each token's final
coordinate axis; tokens and examples never share their statistics. Momentum
is reset at the start of EACH inner sample call, since MAR rounds change the
set of tokens. The base model state_dict and source files are never modified.
"""

import copy
import math
import types
import weakref

import torch


CONFIGS = {
    "baseline": {"method": "original_sample", "effective_cfg": "input"},
    "cfg_original": {"method": "copied_original_cfg", "effective_cfg": "input",
                     "runtime_control_only": True},
    "cfg2": {"method": "cfg", "effective_cfg": 2.0},
    "projection": {"method": "apg", "beta": 0.0, "eta": 0.0,
                   "norm_threshold": 0.0, "effective_cfg": "input"},
    "apg": {"method": "apg", "beta": -0.5, "eta": 0.0,
            "norm_threshold": 0.0, "effective_cfg": "input"},
    "interval": {"method": "interval_cfg", "interval": [0.2, 0.8],
                 "outside_cfg": 1.0, "effective_cfg": "input"},
    "rescale": {"method": "rescale", "alpha": 0.3,
                "std_unbiased": False, "effective_cfg": "input"},
    "apg_interval": {"method": "apg", "beta": -0.5, "eta": 0.0,
                     "norm_threshold": 0.0, "interval": [0.2, 0.8],
                     "outside_cfg": 1.0, "momentum_updates_outside_interval": True,
                     "effective_cfg": "input"},
}
VARIANTS = tuple(name for name in CONFIGS if name != "cfg_original")
EPS = 1e-8


def project_update(update, conditional, eta=0.0):
    """Remove the update parallel to the conditional prediction, token by token."""
    if update.shape != conditional.shape or update.ndim != 2:
        raise ValueError("Projection requires matching [token, coordinate] tensors")
    # Follow the reference APG projection's double-precision arithmetic: small
    # perpendicular updates can otherwise lose precision through cancellation.
    # Only six coordinates are projected; the head remains in its original dtype.
    original_dtype = update.dtype
    update64, conditional64 = update.double(), conditional.double()
    direction = conditional64 / conditional64.norm(dim=-1, keepdim=True).clamp_min(EPS)
    parallel = (update64 * direction).sum(dim=-1, keepdim=True) * direction
    perpendicular = update64 - parallel
    return (perpendicular + eta * parallel).to(original_dtype)


def combine_predictions(conditional, unconditional, cfg, config, t_now,
                        momentum=None):
    """Pure mathematical step; returns (prediction, new momentum, is_guided).

    APG: delta = conditional - unconditional; m = delta + beta * old_m.
    Optional radius clipping follows momentum, then parallel/perpendicular
    decomposition is relative to conditional. The output is c + (cfg-1)*m_mod.
    Interval APG updates momentum even on steps returning c without guidance.
    """
    if conditional.shape != unconditional.shape or conditional.ndim != 2:
        raise ValueError("Guidance requires matching [token, coordinate] predictions")
    method = config["method"]
    effective_cfg = config.get("effective_cfg", "input")
    effective_cfg = cfg if effective_cfg == "input" else float(effective_cfg)
    interval = config.get("interval")
    guided = interval is None or interval[0] <= float(t_now) <= interval[1]
    if method == "apg":
        delta = conditional - unconditional
        if momentum is not None and momentum.shape != delta.shape:
            raise ValueError("APG momentum must match the current token set")
        updated = delta if momentum is None else delta + config["beta"] * momentum
        adjusted = updated
        threshold = config["norm_threshold"]
        if threshold > 0:
            norm = adjusted.norm(dim=-1, keepdim=True)
            adjusted = adjusted * (threshold / norm.clamp_min(EPS)).clamp(max=1.0)
        adjusted = project_update(adjusted, conditional, eta=config["eta"])
        prediction = conditional + (effective_cfg - 1) * adjusted if guided else conditional
        return prediction, updated, bool(guided and effective_cfg != 1.0)
    if not guided:
        return conditional, momentum, False
    # Retain the exact original arithmetic order for cfg_original equivalence.
    prediction = unconditional + effective_cfg * (conditional - unconditional)
    if method == "rescale":
        conditional_std = conditional.std(dim=-1, keepdim=True, unbiased=False)
        prediction_std = prediction.std(dim=-1, keepdim=True, unbiased=False)
        rescaled = prediction * (conditional_std / prediction_std.clamp_min(EPS))
        alpha = config["alpha"]
        prediction = alpha * rescaled + (1 - alpha) * prediction
    return prediction, momentum, bool(effective_cfg != 1.0)


class GuidanceController:
    """Instance-scoped sample replacement and JSON-serializable budget counters."""
    def __init__(self, model, variant):
        if variant not in CONFIGS:
            raise ValueError(f"Unknown guidance variant: {variant}")
        head = model.DiffMLPs
        if hasattr(head, "_research_guidance_controller"):
            raise RuntimeError("Guidance is already installed")
        if int(head.in_channels) != 6:
            raise ValueError("These preregistered experiments require six FSQ coordinates")
        self.variant = variant
        self.config = copy.deepcopy(CONFIGS[variant])
        self.config.update(variant=variant, coordinate_axis=-1, coordinate_count=6,
                           epsilon=EPS, outer_cfg=4.5,
                           projection_dtype="float64; output cast to prediction dtype",
                           parameter_status="preregistered project trial, not a validated motion optimum")
        self._head = weakref.ref(head)
        self._original = head.sample
        self._sample_was_instance = "sample" in head.__dict__
        self._removed = False
        self._inside_sample = False
        self._finite_status = None
        self.reset_stats()

        def count_net(module, inputs, output):
            if self._inside_sample:
                self.stats["net_calls"] += 1
                self.stats["net_token_forwards"] += int(inputs[0].shape[0])
                self._record_finite(output)

        self._hook = head.net.register_forward_hook(count_net)
        # Ordinary controller attribute, never an nn.Module containing its parent.
        head._research_guidance_controller = self

        def sample(owner, z, temperature=1.0, cfg=1.0):
            if self._inside_sample:
                raise RuntimeError("Concurrent or recursive use of one guidance controller is unsupported")
            if not math.isfinite(float(temperature)) or not math.isfinite(float(cfg)):
                raise ValueError("Temperature and CFG must be finite")
            if z.ndim != 2 or (cfg != 1.0 and z.shape[0] % 2):
                raise ValueError("Expected [token, condition] input with paired CFG halves")
            self.stats["sample_calls"] += 1
            self._inside_sample = True
            self._finite_status = torch.isfinite(z).all()
            try:
                if variant == "baseline":
                    # Delegate directly: no copied sampler or arithmetic in baseline.
                    result = self._original(z, temperature, cfg)
                    if cfg != 1.0:
                        self.stats["guided_steps"] += int(owner.num_sampling_steps)
                    else:
                        self.stats["unguided_steps"] += int(owner.num_sampling_steps)
                    self.stats["inner_steps"] += int(owner.num_sampling_steps)
                else:
                    result = self._sample(owner, z, temperature, cfg)
                self._record_finite(result)
                if not bool(self._finite_status):
                    self.stats["nonfinite_samples"] += 1
                    raise FloatingPointError("Nonfinite input, head output, guided prediction, or sampler state")
                self.stats["output_abs_max"] = max(self.stats["output_abs_max"],
                    float(result.detach().abs().max()) if result.numel() else 0.0)
                return result
            finally:
                self._inside_sample = False
                self._finite_status = None

        head.sample = types.MethodType(sample, head)

    def _record_finite(self, tensor):
        # Keep this reduction on-device; synchronize once per complete sample.
        with torch.no_grad():
            self._finite_status = self._finite_status & torch.isfinite(tensor).all()

    def reset_stats(self):
        self.stats = dict(sample_calls=0, net_calls=0, net_token_forwards=0,
                          inner_steps=0, guided_steps=0, unguided_steps=0,
                          momentum_updates=0, nonfinite_samples=0, output_abs_max=0.0)

    def _sample(self, head, z, temperature, cfg):
        # This loop follows DiffTransformer_XPred.sample line by line. Only the
        # conditional/unconditional combination differs for experiment variants.
        device = z.device
        num_steps = head.num_sampling_steps
        if cfg != 1.0:
            batch_size = z.shape[0] // 2
            z_cond = z[:batch_size]
            z_uncond = z[batch_size:]
        else:
            batch_size = z.shape[0]
            z_cond = z
            z_uncond = None
        x_t = torch.randn(batch_size, head.in_channels, device=device) * temperature
        timesteps = torch.linspace(0.02, 0.98, num_steps, device=device)
        x_0 = x_t.clone()
        momentum = None  # A different MAR round must never inherit token momentum.
        for i, t_now in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t_now.item(), device=device)
            if cfg != 1.0:
                x_1_cond = head.net(x_t, t_batch, z_cond)
                x_1_uncond = head.net(x_t, t_batch, z_uncond)
                x_1_pred, momentum, guided = combine_predictions(
                    x_1_cond, x_1_uncond, cfg, self.config, t_now.item(), momentum)
                if self.config["method"] == "apg":
                    self.stats["momentum_updates"] += 1
            else:
                x_1_pred = head.net(x_t, t_batch, z_cond)
                guided = False
            self.stats["inner_steps"] += 1
            self.stats["guided_steps" if guided else "unguided_steps"] += 1
            self._record_finite(x_1_pred)
            if i == num_steps - 1:
                x_t = x_1_pred
            else:
                t_next = timesteps[i + 1]
                if t_now > 0.01:
                    x_0_est = (x_t - t_now * x_1_pred) / (1 - t_now)
                else:
                    x_0_est = x_0
                x_t = t_next * x_1_pred + (1 - t_next) * x_0_est
            self._record_finite(x_t)
        if cfg != 1.0:
            x_t = torch.cat([x_t, x_t], dim=0)
        return x_t

    def remove(self):
        if self._removed:
            return
        head = self._head()
        self._hook.remove()
        if head is not None:
            if self._sample_was_instance:
                head.sample = self._original
            else:
                del head.__dict__["sample"]
            delattr(head, "_research_guidance_controller")
        self._removed = True


def install_guidance(model, variant):
    """Install a sampler-only experiment and return stats/config/remove control."""
    return GuidanceController(model, variant)
