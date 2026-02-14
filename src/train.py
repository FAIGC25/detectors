import typer
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset 
import os
import lightning as pl
from typing import List, Optional
from dotenv import load_dotenv
from src.models.rppg_p_fau_lightning import FauRPPGDeepFakeRecognizer
from src.data.dataset import VideoFolderDataset, split_dataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

app = typer.Typer(pretty_exceptions_show_locals=False)

load_dotenv()

@app.command()
def train(
    config_path: str = typer.Option(
        ...,
        "--config_name", "-c",
        help="Имя .yaml конфига (.yaml)",
    ),
    num_workers: int = typer.Option(4, '--num_workers', "-nw",help='Кол-во воркеров для лоадеров'),
    dataset_paths: List[str] = typer.Option(
        ..., 
        "--dataset_path", "-d", 
        help="Пути к папкам с датасетами (можно указывать несколько раз)"
    ),
    batch_size: int = typer.Option(32, "--batch_size", "-bs",
                                   help='Размер батча'),
    
    load_from_pretrain: Optional[str] = typer.Option(None, "--load_from_pretrain", "-r", help="Путь к .ckpt файлу для возобновления")
):
    """
    Запуск обучения модели FauRPPGDeepFakeRecognizer на нескольких датасетах.
    """

    if load_from_pretrain is not None:
        if not os.path.exists(load_from_pretrain):
            typer.echo(f"❌ Ошибка: Чекпоинт не найден: {load_from_pretrain}")
            raise FileNotFoundError(load_from_pretrain)
        typer.echo(f"🔄 Будет выполнено возобновление из: {load_from_pretrain}")

    if not os.path.exists(config_path):
        config_base_path = os.getenv('EXPERIMENTS_CFG_FOLDER')
        typer.echo(f'[CONFIG BASE PATH] base_path={config_base_path}')
        config_path = os.path.join(config_base_path, config_path)
        
        if not os.path.exists(config_path):
            raise Exception(f'Config not found in {config_base_path}')

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    all_datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            typer.echo(f"📦 Загрузка датасета из: {path}")
            ds = VideoFolderDataset(path, transform=data_transforms)
            all_datasets.append(ds)
        else:
            typer.echo(f"⚠️ Путь не найден и будет пропущен: {path}")

    if not all_datasets:
        raise Exception("Ни один датасет не был загружен. Проверьте пути.")

    full_dataset = ConcatDataset(all_datasets)
    
    base_classes = getattr(all_datasets[0], 'classes', 'Unknown')
    typer.echo(f"Classes (from first DS): {base_classes}")
    typer.echo(f"Total images: {len(full_dataset)}")

    train_ds, val_ds, test_ds = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    typer.echo(f"Split -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True)

    typer.echo(f"📂 Загрузка конфигурации из: {config_path}")


    default_model_cfg = OmegaConf.create({
        'backbone_fau': 'swin_transformer_tiny',
        'num_frames':  128,
        'au_ckpt_path':  './src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth',
        'phys_ckpt_path': './src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth',
        'num_classes':  2,
        'dropout':  0.1,
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

    print("\n🔍 CHECKING TRAINABLE PARAMS:")
    trainable_layers = []
    for name, param in lit_model.model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)

    typer.echo(f"Trainable layers ({len(trainable_layers)}):")
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',       
        filename='best-{epoch:02d}-{val_auc:.4f}', 
        monitor='val_auc',            
        mode='max',                   
        save_top_k=2,                 
        save_last=True,               
        verbose=True
    )


    early_stop_callback = EarlyStopping(
        monitor='val_auc',
        min_delta=0.001,              
        patience=50,                  
        verbose=True,
        mode='max'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    
    trainer = pl.Trainer(callbacks=callbacks,
                         **trainer_cfg)
    typer.echo("🚀 Запуск обучения Lightning...")
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=load_from_pretrain)


if __name__ == "__main__":
    app()