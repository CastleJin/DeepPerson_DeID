from ViTPose_pytorch.models.model import ViTPose
from ViTPose_pytorch.configs.ViTPose_base_coco_256x192 import model as model_cfg
from ViTPose_pytorch.configs.ViTPose_base_coco_256x192 import data_cfg
import torch
import torchvision.transforms as tr
import torch.nn as nn
# from ViTPose_pytorch.models.losses.heatmap_loss import AdaptiveWingLoss
from ViTPose_pytorch.models.losses.mse_loss import JointsMSELoss

def ViT_modules():
    skptpath = "/storage/sjhwang/tr_test/transreid_pytorch/model/vitpose-b-multi-coco.pth"
    img_size = data_cfg['image_size']
    img_transform1 =  tr.Resize((img_size[1], img_size[0]))
    img_transform2 = tr.ToTensor()
    model = ViTPose(model_cfg).cuda()
    load_skpt = torch.load(skptpath)
    model.load_state_dict(load_skpt['state_dict'])
    heatmap_loss = JointsMSELoss()
    return model, img_transform1, img_transform2, heatmap_loss
