import os
import sys
import torch
sys.path.append("..")
import pcn

model_path = "model_pnet_190220/pnet_190220_iter_1499000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "pnet_190220_iter_1499000_.pth")
