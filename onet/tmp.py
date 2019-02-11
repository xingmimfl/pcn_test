import os
import sys
import torch
sys.path.append("..")
import pcn

model_path = "model_onet_190209/onet_190209_iter_1495000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "onet_190209_iter_1495000_.pth")
