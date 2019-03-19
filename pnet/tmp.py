import os
import sys
import torch
sys.path.append("..")
import pcn

model_path = "model_pnet_190310/pnet_190310_iter_1238000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "pnet_190310_iter_1238000_.pth")
