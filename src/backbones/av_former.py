import torch.nn as nn
import torch
from transformers import AutoModel
from .pos import PositionalEncoding
from peft import LoraConfig, get_peft_model

class AVFormer(nn.Module):
    def __init__(self,
                 d_model_out: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6):
        super().__init__()
        self.pe_a = PositionalEncoding(d_model_out)
        self.pe_v = PositionalEncoding(d_model_out)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model_out, nhead=nhead, batch_first=True)
        self.a_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.v_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self,q, z_a, z_v):
        """
        z_a: [Batch, T_audio, D_audio_in]
        z_v: [Batch, T_video, D_video_in]
        """
        feat_a = self.pe_a(z_a)
        feat_v = self.pe_v(z_v)
        out_a = self.a_transformer(q, feat_a)
        out_v = self.v_transformer(q, feat_v)

        return out_a, out_v

class QSharedFormer(nn.Module):
    def __init__(self,
                 d_model: int = 512):
        super().__init__()
        self.a_attn = nn.MultiheadAttention(d_model,num_heads=1)
        self.v_attn = nn.MultiheadAttention(d_model,num_heads=1)

    def forward(self,q, z_a, z_v):
        out_a, _ = self.a_attn(q, z_a, z_a)
        out_v, _ = self.v_attn(q, z_v, z_v)

        return out_a, out_v


class AVFormerPretrained(nn.Module):
    def __init__(self,
                 model_name: str,
                 lora_cfg: dict | None,
                 embed_dim: int = 512):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        if lora_cfg:
            peft_config = LoraConfig(**lora_cfg)
            self.model = get_peft_model(self.model, peft_config)

        self.hidden_size = self.model.config.hidden_size
        self.input_proj = nn.Identity() if self.hidden_size ==  embed_dim else \
                          nn.Linear(embed_dim, self.hidden_size)

    def forward(self,q, z_features):
        embed = torch.cat([q, z_features], dim=1)
        embed = self.input_proj(embed)

        attention_mask = torch.ones(embed.shape[:2], device=embed.device, dtype=torch.long)
        return self.model(inputs_embeds=embed,
                          attention_mask=attention_mask,
                          output_hidden_states=True)


class MixedFormer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6):
        super().__init__()
        self.audio_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.video_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.pe = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


    def forward(self, q, z_a, z_v):
        """
        z_mixed = torch.cat([z_mixed_a,z_mixed_v], dim = 1)

        z_mixed_a = Att(q=z_a, k=z_v,v = k=z_v)
        z_mixed_v = Att(q=z_v, k=z_a,v = k=z_a)
        z_* = proj(Z_*)

        """
        z_a = z_a + self.audio_type_embed
        z_v = z_v + self.video_type_embed

        z_a = self.pe(z_a)
        z_v = self.pe(z_v)

        z_mixed = torch.cat([z_a, z_v], dim=1)

        out_av = self.transformer(tgt=q, memory=z_mixed)

        return out_av


class QSharedMixedFormer(nn.Module):
    def __init__(self,
                 d_model: int = 512):
        super().__init__()
        self.audio_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.video_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.pe = PositionalEncoding(d_model)
        self.attn = nn.MultiheadAttention(d_model,num_heads=1)



    def forward(self, q, z_a, z_v):
        """
        z_mixed = torch.cat([z_mixed_a,z_mixed_v], dim = 1)

        z_mixed_a = Att(q=z_a, k=z_v,v = k=z_v)
        z_mixed_v = Att(q=z_v, k=z_a,v = k=z_a)
        z_* = proj(Z_*)

        """
        z_a = z_a + self.audio_type_embed
        z_v = z_v + self.video_type_embed

        z_a = self.pe(z_a)
        z_v = self.pe(z_v)

        z_mixed = torch.cat([z_a, z_v], dim=1)

        out_av, _ = self.attn(query=q, key=z_mixed, value=z_mixed)

        return out_av