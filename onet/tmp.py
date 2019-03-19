import os
import sys
import torch
sys.path.append("..")
import pcn

model_path = "model_onet_190318/onet_190318_iter_1499000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "onet_190227_iter_1499000_.pth")
