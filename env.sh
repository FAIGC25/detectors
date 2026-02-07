#!/bin/bash

CHECKPOINT_DIR="src/backbones/MEGraphAU/checkpoints"
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${CYAN}=== ME-GraphAU Weight Manager ===${NC}"

echo "Что вы хотите скачать?"
echo "1) Базовые предобученные Backbones (PyTorch/Swin)"
echo "2) Полностью обученные FAU модели (Google Drive)"
read -p "Выберите [1-2]: " TYPE_CHOICE

if [ "$TYPE_CHOICE" == "1" ]; then
    echo -e "\n${GREEN}Доступные Backbones:${NC}"
    echo "resnet18, resnet34, resnet50, resnet101, resnet152"
    echo "swin-tiny, swin-small, swin-base"
    read -p "Введите название модели: " MODEL_NAME
    python3 load.py backbone "$MODEL_NAME"

elif [ "$TYPE_CHOICE" == "2" ]; then
    echo -e "\n${GREEN}Выберите датасет FAU:${NC}"
    echo "1) BP4D"
    echo "2) DISFA"
    read -p "Выбор [1-2]: " DS_CHOICE

    CAT="fau-bp4d"
    [ "$DS_CHOICE" == "2" ] && CAT="fau-disfa"

    echo -e "\n${GREEN}Доступные архитектуры:${NC}"
    if [ "$DS_CHOICE" == "1" ]; then echo "swin-tiny, swin-small, swin-base, resnet50, resnet101"; else echo "resnet50, swin-base"; fi

    read -p "Введите название: " MODEL_NAME
    python3 load.py "$CAT" "$MODEL_NAME"
fi

echo -e "\n${CYAN}Готово! Файлы в $CHECKPOINT_DIR${NC}"