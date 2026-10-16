"""Synthetic forward/backward checks; no real datasets or checkpoints."""
import argparse
import json
import platform
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from nilm_lab.models import make_model

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["NILMFormer", "FCN", "BERT4NILM", "SGN"])
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args = p.parse_args()
torch.set_num_threads(1)
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but no accessible GPU/compatible driver")
arms = [("R", 0), ("O", 2), ("B", 6), ("BP", 6), ("P", 6)] if args.model in {"NILMFormer", "FCN"} else ([("R", 0)] if args.model == "BERT4NILM" else [("O", 2)])
for arm, classes in arms:
    torch.manual_seed(7)
    model = make_model(args.model, 64, classes).to(args.device)
    x = torch.randn(2, 9, 64, device=args.device)
    power, logits, activity = model(x)
    assert power.shape == (2, 64), power.shape
    loss = power.square().mean()
    if classes:
        assert logits.shape == (2, classes, 64), logits.shape
        loss = loss + torch.nn.functional.cross_entropy(logits, torch.zeros(2,64,dtype=torch.long,device=args.device))
    assert torch.isfinite(loss)
    loss.backward()
    grads = [v.grad for v in model.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    torch.optim.Adam(model.parameters(), lr=1e-4).step()
print(json.dumps({"model":args.model,"arms":[a for a,c in arms],"python":platform.python_version(),"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"device":args.device,"result":"PASS"}, indent=2))
