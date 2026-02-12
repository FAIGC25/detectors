from src.backbones.MEGraphAU.model.MEFL import MEFARG
from src.backbones.MEGraphAU.model.ANFL import MEFARG as ANFARG
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

class FAUEncoderANFL(ANFARG):
    def __init__(self, num_classes=12, backbone='swin_transformer_base',neighbor_num=4, metric='dots'):
        super().__init__(num_classes=num_classes, backbone=backbone, neighbor_num=neighbor_num, metric=metric)


    def load_pretrained(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"❌ Файл не найден: {checkpoint_path}")
            return False

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"✅ Веса загружены из {checkpoint_path}")
        print(f"🔹 Пропущен слой (нормально): {missing}")
        return True

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_linear(x)
        f_u = []
        for i, layer in enumerate(self.head.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        f_v = self.head.gnn(f_v)
        return f_v

class FAUEncoder(MEFARG):
    def __init__(self, num_classes=12, backbone='swin_transformer_base'):
        super().__init__(num_classes=num_classes, backbone=backbone)
        self.num_classes = num_classes


    def load_pretrained(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"❌ Файл не найден: {checkpoint_path}")
            return False

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"✅ Веса загружены из {checkpoint_path}")
        print(f"🔹 Пропущен слой (нормально): {missing}")
        return True

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_linear(x)
        f_u = []
        for i, layer in enumerate(self.head.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        f_e = self.head.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        f_v, f_e = self.head.gnn(f_v, f_e)

        b, n, c = f_v.shape
        sc = self.head.sc
        sc = self.head.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1, keepdim=False)
        cl_edge = self.head.edge_fc(f_e)
        return f_v, cl, cl_edge


if __name__ == '__main__':

    MODEL_TYPE = 'resnet50'
    NUM_AU = 8
    CHECKPOINT = './src/backbones/MEGraphAU/checkpoints/MEFARG_resnet50_DISFA_fold2.pth'


    embed = FAUEncoder(backbone=MODEL_TYPE, num_classes=NUM_AU)
    if embed.load_pretrained(CHECKPOINT):
        test_input = torch.rand(1, 3, 224, 224)
        output = embed(test_input)
        print(f"Выходной размер: {output.shape}")