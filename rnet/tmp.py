import os
import sys
import torch
sys.path.append("..")
import pcn

model_path = "model_pnet_190312/pnet_190312_iter_979000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "pnet_190312_iter_979000_.pth")
