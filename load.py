import typer
import gdown
import os
import zipfile
import subprocess

app = typer.Typer()

# База ссылок
WEIGHTS_DB = {
    "backbone": {
        "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        "swin-tiny": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        "swin-small": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        "swin-base": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
    },
    "fau-bp4d": {
        "resnet50": "1EiQd6q7x1bEO6JBLi3s2y5348EuVdP3L",
        "resnet101": "1Ti0auMA5o94toJfszuHoMlSlWUumm9L8",
        "swin-tiny": "1i-ra0dtoEhwIep6goZ55PvEgwE3kecbl",
        "swin-small": "1BT4n7_5Wr6bGxHWVf3WrT7uBT0Zg9B5c",
        "swin-base": "1gYVHRjIj6ounxtTdXMje4WNHFMfCTVF9",
    },
    "fau-disfa": {
        "resnet50": "1V-imbmhg-OgcP2d9SETT5iswNtCA0f8_",
        "swin-base": "1T44KPDaUhi4J_C-fWa6RxXNkY3yoDwIi",
    }
}

def download_file(url_or_id, output_dir, is_gdrive=False):
    if not is_gdrive:
        # wget -nc не перекачивает файл, если он уже есть
        subprocess.run(["wget", "-nc", url_or_id, "-P", output_dir])
    else:
        # gdown скачивает с Google Drive
        out_path = gdown.download(f'https://drive.google.com/uc?id={url_or_id}', 
                                  output=output_dir + "/", quiet=False, fuzzy=True)
        return out_path
    return None

@app.command()
def download(category: str, model: str):
    output_dir = "src/backbones/MEGraphAU/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    if category.startswith("fau"):
        if model in WEIGHTS_DB["backbone"]:
            typer.secho(f"📦 Авто-загрузка базового backbone для {model}...", fg="blue")
            download_file(WEIGHTS_DB["backbone"][model], output_dir, is_gdrive=False)

    if category not in WEIGHTS_DB or model not in WEIGHTS_DB[category]:
        typer.secho(f"❌ Ошибка: {model} не найден в {category}", fg="red")
        return

    target = WEIGHTS_DB[category][model]

    if category == "backbone":
        download_file(target, output_dir, is_gdrive=False)
    else:
        typer.secho(f"🔥 Скачиваю FAU Model ({category}): {model}...", fg="yellow")
        out_path = download_file(target, output_dir, is_gdrive=True)
        if out_path and out_path.endswith('.zip'):
            with zipfile.ZipFile(out_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(out_path)
            typer.secho(f"✅ Фолды для {model} распакованы.", fg="green")

if __name__ == "__main__":
    app()