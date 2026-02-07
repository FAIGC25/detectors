# ====================================================================================================================
# Using Audio Model. Dont use
# ====================================================================================================================


from transformers import VideoMAEModel, VideoMAEImageProcessor
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch.nn as nn
import torch
from src.backbones.fau_encoder import FAUEncoder



class BaseAVDFModule(nn.Module):
    def __init__(self, d_model: int = 512, num_latents: int = 32,
                 backbone_fau='resnet50', au_ckpt_path='checkpoints/fau_resnet50.pth',
                 num_fau_classes: int = 12,
                 video_model_name = "MCG-NJU/videomae-base",
                 audio_model_name = "facebook/hubert-base-ls960"):
        super().__init__()

        self.d_model = d_model
        self.num_latents = num_latents

        self.v_encoder = VideoMAEModel.from_pretrained(video_model_name)
        for param in self.v_encoder.parameters():
            param.requires_grad = False

        self.fau_encoder = FAUEncoder(num_classes=num_fau_classes,
                                      backbone=backbone_fau)

        if au_ckpt_path:
            self.fau_encoder.load_pretrained(au_ckpt_path)

        for param in self.fau_encoder.parameters():
            param.requires_grad = False

        self.v_proj = nn.Linear(self.v_encoder.config.hidden_size, d_model)
        self.fau_proj = nn.Linear(self.fau_encoder.out_channels, d_model)

        self.a_encoder = HubertModel.from_pretrained(audio_model_name)
        self.a_proj = nn.Linear(self.a_encoder.config.hidden_size, d_model)

        self.q_learnable = nn.Parameter(torch.randn(1, num_latents, d_model))

    def extract_features(self, video_pixel_values, audio_input_values):
        B, C, T, H, W = video_pixel_values.shape


        outputs_a = self.a_encoder(audio_input_values)
        z_a = outputs_a.last_hidden_state
        z_a = self.a_proj(z_a)


        with torch.no_grad():
            mae_input = video_pixel_values.permute(0, 2, 1, 3, 4)
            outputs_v = self.v_encoder(mae_input)
            z_v_mae = outputs_v.last_hidden_state
            z_v_mae = self.v_proj(z_v_mae)

        fau_input = video_pixel_values.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        z_v_fau = self.fau_encoder(fau_input)
        z_v_fau = z_v_fau.view(B, T, -1, z_v_fau.shape[-1]).flatten(1, 2)
        z_v_fau = self.fau_proj(z_v_fau)
        z_v = torch.cat([z_v_mae, z_v_fau], dim=1)

        q = self.q_learnable.expand(B, -1, -1)

        return q, z_a, z_v