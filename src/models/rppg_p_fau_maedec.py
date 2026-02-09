# ====================================================================================================================
# VideoMAe as Decoder for FAU and rPPG features
# ====================================================================================================================


import torch
import torch.nn as nn
from transformers import VideoMAEModel
from src.backbones.fau_encoder import FAUEncoder
from src.backbones.rppg_encoder import RPPGEncoder
from src.backbones.pos import PositionalEncoding
from src.pooler.attn_pooler import AttentionPooler
from peft import LoraConfig, get_peft_model

class DeepfakeDetector(nn.Module):
    def __init__(self,
                 videomae_model_name: str ='MCG-NJU/videomae-base',
                 backbone_fau: str = 'resnet50',
                 num_au_classes: int = 8,
                 au_ckpt_path: str | None = './src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth',
                 phys_ckpt_path: str | None = './src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth',
                 num_classes: int = 2,
                 dropout:int = 0.1,
                 lora_cfg: dict | None = None):
        super().__init__()

        self.au_encoder = FAUEncoder(num_classes=num_au_classes, backbone=backbone_fau)
        if au_ckpt_path:
            self.au_encoder.load_pretrained(au_ckpt_path)
        self.phys_encoder = RPPGEncoder(frames=16)

        if phys_ckpt_path:
            self.phys_encoder.load_pretrained(phys_ckpt_path)


        self.videomae = VideoMAEModel.from_pretrained(videomae_model_name)
        if lora_cfg:
            peft_config = LoraConfig(**lora_cfg)
            self.videomae = get_peft_model(self.videomae, peft_config)

        self.au_proj = nn.Linear(self.au_encoder.out_channels, self.videomae.config.hidden_size)
        self.phys_proj = nn.Linear(self.phys_encoder.out_channels, self.videomae.config.hidden_size)
        self.segment_embed = nn.Embedding(2, self.videomae.config.hidden_size)

        self.pos = PositionalEncoding(self.videomae.config.hidden_size)

        self.norm = nn.LayerNorm(self.videomae.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.videomae.config.hidden_size, num_classes)
        self.attn_pooler = AttentionPooler(self.videomae.config.hidden_size)


    def forward(self, x_video, return_attention=False):
        """
        x_video: [B, 3, T=16, 224, 224]
        """
        B, C, T, H, W = x_video.shape
        device = x_video.device

        x_au_input = x_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        au_raw = self.au_encoder(x_au_input)
        tokens_au = self.au_proj(au_raw)
        tokens_au = tokens_au.view(B, T, -1, tokens_au.shape[-1]).flatten(1,2)

        tokens_au = tokens_au + self.segment_embed(torch.tensor(0, device=device))
        tokens_au = self.pos(tokens_au)
        _, phys_raw = self.phys_encoder(x_video)
        tokens_phys = self.phys_proj(phys_raw)
        tokens_phys = tokens_phys + self.segment_embed(torch.tensor(1, device=device))
        tokens_phys = self.pos(tokens_phys)
        combined_embeddings = torch.cat([tokens_au, tokens_phys], dim=1)

        encoder_outputs = self.videomae.encoder(combined_embeddings)
        last_hidden_state = encoder_outputs.last_hidden_state
        #features = torch.mean(last_hidden_state, dim=1)
        features, attn_weights = self.attn_pooler(last_hidden_state)
        logits = self.classifier(self.dropout(self.norm(features)))
        if return_attention:
            return logits, attn_weights
        return logits

if __name__ == '__main__':
    # Пример инициализации
    try:
        model = DeepfakeDetector(
            videomae_model_name='MCG-NJU/videomae-base',
            num_au_classes=8,
            lora_cfg={"inference_mode":False,
            'r':8,
            "lora_alpha":32,
            "lora_dropout":0.1,
            "target_modules": ["query", "value", "key"]}
        )
        dummy_input = torch.randn(2, 3, 16, 224, 224)
        print("Запуск forward pass с VideoMAE-LoRA в роли декодера...")
        print(model)
        output = model(dummy_input)
        print(f"✅ Успех! Логиты: {output.shape}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()