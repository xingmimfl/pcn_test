import os
import sys
import torch
sys.path.append("..")
import pcn

model_path = "model_pnet_190219/pnet_190219_iter_1999000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "pnet_190219_iter_1999000_.pth")
