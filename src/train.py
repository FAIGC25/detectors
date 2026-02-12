import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from models.rppg_p_fau_maedec import DeepfakeDetector
from loss.contrastive import InfoNCEConsistencyLoss



class LitDeepfakeDetector(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        T_max: int = 10,
        lambda_nce: float = 0.5,
        num_classes: int = 2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepfakeDetector(**model_params)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nce = InfoNCEConsistencyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_prec = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_rec = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        # --- МЕТРИКИ (Validation) ---
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_prec = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_rec = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, return_info_nce=True)
        logits = output["logits"]
        nce_logits = output["nce_logits"]
        attn_weights = output["attn_weights"]

        loss_ce = self.criterion_ce(logits, y)

        loss_nce = self.criterion_nce(nce_logits, attn_weights)

        total_loss = loss_ce + (self.hparams.lambda_nce * loss_nce)

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ce", loss_ce, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_nce", loss_nce, prog_bar=False, on_step=True, on_epoch=True)

        probs = F.softmax(logits, dim=1)

        acc = self.train_acc(logits, y)
        f1 = self.train_f1(logits, y)
        prec = self.train_prec(logits, y)
        rec = self.train_rec(logits, y)
        auc = self.train_auc(probs, y)

        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", f1, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_auc", auc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_prec", prec, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_rec", rec, prog_bar=False, on_step=False, on_epoch=True)


        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, return_info_nce=False)
        logits = output
        val_loss = self.criterion_ce(logits, y)
        probs = F.softmax(logits, dim=1)

        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        prec = self.val_prec(logits, y)
        rec = self.val_rec(logits, y)
        auc = self.val_auc(probs, y)


        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_prec", prec, prog_bar=False)
        self.log("val_rec", rec, prog_bar=False)
        self.log("val_auc", auc, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.T_max, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

# --- ПРИМЕР ЗАПУСКА ---
if __name__ == "__main__":
    # Параметры для твоей модели
    model_config = {
        "videomae_model_name": 'MCG-NJU/videomae-base',
        "num_au_classes": 12,
        "num_frames": 16,
        "lora_cfg": {
            "inference_mode": False,
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["query", "value", "key", "dense"]
        }
    }


    lit_model = LitDeepfakeDetector(
        model_params=model_config,
        lr=1e-4,
        T_max=10,
        lambda_nce=0.5
    )


    dummy_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 16, 224, 224), torch.tensor(1)) for _ in range(10)],
        batch_size=2
    )

    trainer = pl.Trainer(max_epochs=2, accelerator="auto", devices=1)
    print("🚀 Запуск обучения Lightning...")
    trainer.fit(lit_model, dummy_loader)