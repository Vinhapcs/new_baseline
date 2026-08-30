import math
import torch
from torch import nn
import torch.nn.functional as F

class ImplicitMetricField(nn.Module):
    def __init__(self, center, scale, fourier_levels=6, hidden=96, layers=4):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float32).reshape(1, 3)
        scale = torch.as_tensor(scale, dtype=torch.float32).reshape(1, 1)
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.fourier_levels = int(fourier_levels)
        self.hidden = int(hidden)
        self.layers = int(layers)

        in_dim = 3 * (1 + 2 * self.fourier_levels)
        blocks = []
        d = in_dim
        for _ in range(self.layers):
            blocks += [nn.Linear(d, self.hidden), nn.SiLU()]
            d = self.hidden
        self.trunk = nn.Sequential(*blocks)
        self.normal_head = nn.Linear(self.hidden, 3)
        self.trust_head = nn.Linear(self.hidden, 1)

    def normalize_world(self, x):
        return (x - self.center) / self.scale.clamp_min(1e-8)

    def encode(self, x):
        feats = [x]
        for k in range(self.fourier_levels):
            f = (2.0 ** k) * math.pi
            feats.append(torch.sin(f * x))
            feats.append(torch.cos(f * x))
        return torch.cat(feats, dim=-1)

    def forward_normalized(self, xn, return_logit=False):
        h = self.trunk(self.encode(xn))
        normal = F.normalize(self.normal_head(h), dim=-1, eps=1e-8)
        trust_logit = self.trust_head(h)
        if return_logit:
            return normal, trust_logit
        return normal, torch.sigmoid(trust_logit)

    def forward(self, x_world):
        return self.forward_normalized(self.normalize_world(x_world), return_logit=False)

    def config_dict(self):
        return {
            "fourier_levels": self.fourier_levels,
            "hidden": self.hidden,
            "layers": self.layers,
        }

def load_metric_field(path, device="cuda"):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    field = ImplicitMetricField(
        center=ckpt["center"],
        scale=ckpt["scale"],
        **ckpt["config"],
    )
    field.load_state_dict(ckpt["state_dict"])
    field.eval().to(device)
    for p in field.parameters():
        p.requires_grad_(False)
    return field

def metric_strength(iteration, start, end, ramp):
    if iteration < start or iteration > end:
        return 0.0
    if ramp <= 0:
        return 1.0
    up = min(max((iteration - start) / float(ramp), 0.0), 1.0)
    down = min(max((end - iteration) / float(ramp), 0.0), 1.0)
    return min(up, down)

@torch.no_grad()
def transform_xyz_gradient_inplace(gaussians, field, strength, rho_min=0.05, chunk_size=65536):
    grad = gaussians._xyz.grad
    if grad is None or strength <= 0:
        return {"active": 0, "normal_ratio_before": 0.0, "normal_ratio_after": 0.0, "suppression_ratio": 1.0, "mean_trust": 0.0, "mean_rho": 1.0}

    gnorm = torch.linalg.vector_norm(grad, dim=-1)
    active_idx = torch.nonzero(gnorm > 1e-12, as_tuple=False).squeeze(-1)
    if active_idx.numel() == 0:
        return {"active": 0, "normal_ratio_before": 0.0, "normal_ratio_after": 0.0, "suppression_ratio": 1.0, "mean_trust": 0.0, "mean_rho": 1.0}

    normal_before_sum, normal_after_sum, total_grad_sum, trust_sum, rho_sum, total = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    for lo in range(0, active_idx.numel(), int(chunk_size)):
        ids = active_idx[lo:lo + int(chunk_size)]
        x = gaussians._xyz.detach()[ids]
        g = grad[ids]

        n, trust = field(x)
        gn_scalar = (g * n).sum(dim=-1, keepdim=True)
        g_normal = gn_scalar * n
        g_tangent = g - g_normal

        rho = 1.0 - float(strength) * trust * (1.0 - float(rho_min))
        g_new = g_tangent + rho * g_normal
        grad[ids] = g_new

        normal_before = torch.linalg.vector_norm(g_normal, dim=-1)
        normal_after = torch.linalg.vector_norm(rho * g_normal, dim=-1)
        total_grad = torch.linalg.vector_norm(g, dim=-1)

        normal_before_sum += float(normal_before.sum().item())
        normal_after_sum += float(normal_after.sum().item())
        total_grad_sum += float(total_grad.sum().item())
        trust_sum += float(trust.sum().item())
        rho_sum += float(rho.sum().item())
        total += int(ids.numel())

    denom = max(total_grad_sum, 1e-12)
    return {
        "active": total,
        "normal_ratio_before": normal_before_sum / denom,
        "normal_ratio_after": normal_after_sum / denom,
        "suppression_ratio": normal_after_sum / max(normal_before_sum, 1e-12),
        "mean_trust": trust_sum / max(total, 1),
        "mean_rho": rho_sum / max(total, 1),
    }

@torch.no_grad()
def transform_xyz_gradient_evidence_inplace(gaussians, field, strength, rho_min=0.05, trust_threshold=0.78, trust_temperature=0.08, normal_need_beta=0.35, normal_need_clip=3.0, chunk_size=65536):
    grad = gaussians._xyz.grad
    empty = {"active": 0, "normal_ratio_before": 0.0, "normal_ratio_after": 0.0, "suppression_ratio": 1.0, "mean_geo_trust": 0.0, "mean_calibrated_trust": 0.0, "mean_effective_trust": 0.0, "mean_normal_need": 0.0, "mean_rho": 1.0}
    if grad is None or strength <= 0:
        return empty

    gnorm = torch.linalg.vector_norm(grad, dim=-1)
    active_idx = torch.nonzero(gnorm > 1e-12, as_tuple=False).squeeze(-1)
    if active_idx.numel() == 0:
        return empty

    normal_abs_chunks = []
    for lo in range(0, active_idx.numel(), int(chunk_size)):
        ids = active_idx[lo:lo + int(chunk_size)]
        x = gaussians._xyz.detach()[ids]
        g = grad[ids]
        n, _ = field(x)
        normal_abs_chunks.append(torch.abs((g * n).sum(dim=-1)))
    normal_mean = torch.cat(normal_abs_chunks).mean().clamp_min(1e-12)

    normal_before_sum, normal_after_sum, total_grad_sum, geo_sum, cal_sum, eff_sum, need_sum, rho_sum, total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    temp = max(float(trust_temperature), 1e-4)

    for lo in range(0, active_idx.numel(), int(chunk_size)):
        ids = active_idx[lo:lo + int(chunk_size)]
        x = gaussians._xyz.detach()[ids]
        g = grad[ids]

        n, q_geo = field(x)
        gn_scalar = (g * n).sum(dim=-1, keepdim=True)
        g_normal = gn_scalar * n
        g_tangent = g - g_normal

        q_cal = torch.sigmoid((q_geo - float(trust_threshold)) / temp)
        normal_need = torch.abs(gn_scalar) / normal_mean
        normal_need = torch.clamp(normal_need, min=0.0, max=float(normal_need_clip))
        correction_gate = torch.exp(-float(normal_need_beta) * normal_need)
        q_eff = q_cal * correction_gate

        rho = 1.0 - float(strength) * q_eff * (1.0 - float(rho_min))
        grad[ids] = g_tangent + rho * g_normal

        nb = torch.linalg.vector_norm(g_normal, dim=-1)
        na = torch.linalg.vector_norm(rho * g_normal, dim=-1)
        tg = torch.linalg.vector_norm(g, dim=-1)

        normal_before_sum += float(nb.sum().item())
        normal_after_sum += float(na.sum().item())
        total_grad_sum += float(tg.sum().item())
        geo_sum += float(q_geo.sum().item())
        cal_sum += float(q_cal.sum().item())
        eff_sum += float(q_eff.sum().item())
        need_sum += float(normal_need.sum().item())
        rho_sum += float(rho.sum().item())
        total += int(ids.numel())

    return {
        "active": total,
        "normal_ratio_before": normal_before_sum / max(total_grad_sum, 1e-12),
        "normal_ratio_after": normal_after_sum / max(total_grad_sum, 1e-12),
        "suppression_ratio": normal_after_sum / max(normal_before_sum, 1e-12),
        "mean_geo_trust": geo_sum / max(total, 1),
        "mean_calibrated_trust": cal_sum / max(total, 1),
        "mean_effective_trust": eff_sum / max(total, 1),
        "mean_normal_need": need_sum / max(total, 1),
        "mean_rho": rho_sum / max(total, 1),
    }
