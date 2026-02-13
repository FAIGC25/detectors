import typer
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import lightning as pl
from dotenv import load_dotenv
from src.models.rppg_p_fau_lightning import FauRPPGDeepFakeRecognizer
from src.data.dataset import VideoFolderDataset, split_dataset

app = typer.Typer(pretty_exceptions_show_locals=False)

load_dotenv()

@app.command()
def train(
    config_path: str = typer.Option(
        ...,
        "--config_name", "-c",
        help="Имя .yaml конфига (.yaml)",
    ),
    batch_size: int = typer.Option(32, "--batch_size", "-bs",
                                   help='Размер батча')
):
    """
    Запуск обучения модели FauRPPGDeepFakeRecognizer.
    """

    if not os.path.exists(config_path):
        config_base_path = os.getenv('EXPERIMENTS_CFG_FOLDER')
        typer.echo(f'[CONFIG BASE PATH] base_path={config_base_path}')
        config_path = os.path.join(config_base_path, config_path)
        typer.echo(f'[PATH] path={config_path}')

        if not os.path.exists(config_path):
            raise Exception(f'Select config names from {config_base_path}. If config is not exist please make .yaml config and try again')

    dataset_path = os.getenv('DATASET_PATH')
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_dataset = VideoFolderDataset(dataset_path, transform=data_transforms)

    typer.echo(f"Classes: {full_dataset.classes}")
    typer.echo(f"Len images: {len(full_dataset)}")

    train_ds, val_ds, test_ds = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    typer.echo(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    typer.echo(f"📂 Загрузка конфигурации из: {config_path}")


    default_model_cfg = OmegaConf.create({
        'backbone_fau': 'swin_transformer_tiny',
        'num_frames':  16,
        'au_ckpt_path':  './src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth',
        'phys_ckpt_path': './src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth',
        'num_classes':  2,
        'dropout':  0.1,
        'temperature': 0.07,
        "videomae_model_name": 'MCG-NJU/videomae-base',
        "num_au_classes": 12,
        "lora_cfg": {
            "inference_mode": False,
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["query", "value", "key"]
        }
    })

    file_config = OmegaConf.load(config_path)
    
    model_cfg = file_config.model_params
    train_cfg = file_config.train_params
    trainer_cfg = file_config.trainer_params
    
    final_config = OmegaConf.merge(model_cfg, default_model_cfg)
    model_cfg = OmegaConf.to_container(final_config, resolve=True)

    
    lit_model = FauRPPGDeepFakeRecognizer(
        model_params=model_cfg,
        **train_cfg)


    trainer = pl.Trainer(**trainer_cfg)
    typer.echo("🚀 Запуск обучения Lightning...")
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    app()