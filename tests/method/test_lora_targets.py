"""plearn/method/estimator.py add_lora on a toy Qwen3.5-like layer (CPU): the default names (200 modules on the 4B) and
all_linear (every linear layer, with the gated delta-net's linear_attn in_proj_a and in_proj_b)."""
import unittest

import torch


class LinearAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for n, (o, i) in {"in_proj_qkv": (24, 8), "in_proj_z": (16, 8), "in_proj_b": (2, 8), "in_proj_a": (2, 8), "out_proj": (8, 16)}.items():
            setattr(self, n, torch.nn.Linear(i, o, bias=False))
        self.conv1d = torch.nn.Conv1d(24, 24, 4, groups=24)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = LinearAttn(); self.mlp = torch.nn.ModuleDict({k: torch.nn.Linear(8, 8, bias=False) for k in ("gate_proj", "up_proj", "down_proj")})
        self.lm_head = torch.nn.Linear(8, 50, bias=False)


class LoraTargetTests(unittest.TestCase):
    def names(self, m):
        from plearn.method.estimator import LoRALinear
        return sorted(n.rsplit(".", 1)[-1] for n, x in m.named_modules() if isinstance(x, LoRALinear))

    def test_default_leaves_the_gates(self):
        from plearn.method.estimator import add_lora
        m = Toy(); self.assertEqual(add_lora(m, r=4), 6)
        self.assertEqual(self.names(m), ["down_proj", "gate_proj", "in_proj_qkv", "in_proj_z", "out_proj", "up_proj"])

    def test_all_linear_adds_the_gated_deltanet_gates(self):
        from plearn.method.estimator import add_lora
        m = Toy(); self.assertEqual(add_lora(m, r=4, all_linear=True), 8)
        self.assertEqual(self.names(m), ["down_proj", "gate_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj", "up_proj"])
        self.assertIsInstance(m.lm_head, torch.nn.Linear); self.assertIsInstance(m.linear_attn.conv1d, torch.nn.Conv1d)   # lm_head, conv1d: frozen
        self.assertTrue(all(not p.requires_grad for n, p in m.named_parameters() if "lm_head" in n or "conv1d" in n))


if __name__ == "__main__":
    unittest.main()
