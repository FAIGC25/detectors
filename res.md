.
├── env.sh
├── main.py
├── pyproject.toml
├── README.md
├── res.md
├── src
│   ├── backbones
│   │   ├── av_former.py
│   │   ├── fau_encoder.py
│   │   ├── MEGraphAU
│   │   │   ├── checkpoints
│   │   │   │   ├── checkpoints.txt
│   │   │   │   ├── resnet18-5c106cde.pth
│   │   │   │   └── swin_base_patch4_window7_224_22k.pth
│   │   │   ├── conf.py
│   │   │   ├── config
│   │   │   │   ├── BP4D_config.yaml
│   │   │   │   └── DISFA_config.yaml
│   │   │   ├── data
│   │   │   │   ├── BP4D
│   │   │   │   │   ├── img
│   │   │   │   │   │   └── F001
│   │   │   │   │   │       └── T1
│   │   │   │   │   │           └── 2440.jpg
│   │   │   │   │   └── list
│   │   │   │   │       ├── BP4D_test_img_path_fold1.txt
│   │   │   │   │       ├── BP4D_test_img_path_fold2.txt
│   │   │   │   │       ├── BP4D_test_img_path_fold3.txt
│   │   │   │   │       ├── BP4D_test_label_fold1.txt
│   │   │   │   │       ├── BP4D_test_label_fold2.txt
│   │   │   │   │       ├── BP4D_test_label_fold3.txt
│   │   │   │   │       ├── BP4D_train_AU_relation_fold1.txt
│   │   │   │   │       ├── BP4D_train_AU_relation_fold2.txt
│   │   │   │   │       ├── BP4D_train_AU_relation_fold3.txt
│   │   │   │   │       ├── BP4D_train_img_path_fold1.txt
│   │   │   │   │       ├── BP4D_train_img_path_fold2.txt
│   │   │   │   │       ├── BP4D_train_img_path_fold3.txt
│   │   │   │   │       ├── BP4D_train_label_fold1.txt
│   │   │   │   │       ├── BP4D_train_label_fold2.txt
│   │   │   │   │       ├── BP4D_train_label_fold3.txt
│   │   │   │   │       ├── BP4D_weight_fold1.txt
│   │   │   │   │       ├── BP4D_weight_fold2.txt
│   │   │   │   │       └── BP4D_weight_fold3.txt
│   │   │   │   └── DISFA
│   │   │   │       ├── img
│   │   │   │       │   └── SN001
│   │   │   │       │       └── 0.png
│   │   │   │       └── list
│   │   │   │           ├── DISFA_test_img_path_fold1.txt
│   │   │   │           ├── DISFA_test_img_path_fold2.txt
│   │   │   │           ├── DISFA_test_img_path_fold3.txt
│   │   │   │           ├── DISFA_test_label_fold1.txt
│   │   │   │           ├── DISFA_test_label_fold2.txt
│   │   │   │           ├── DISFA_test_label_fold3.txt
│   │   │   │           ├── DISFA_train_AU_relation_fold1.txt
│   │   │   │           ├── DISFA_train_AU_relation_fold2.txt
│   │   │   │           ├── DISFA_train_AU_relation_fold3.txt
│   │   │   │           ├── DISFA_train_img_path_fold1.txt
│   │   │   │           ├── DISFA_train_img_path_fold2.txt
│   │   │   │           ├── DISFA_train_img_path_fold3.txt
│   │   │   │           ├── DISFA_train_label_fold1.txt
│   │   │   │           ├── DISFA_train_label_fold2.txt
│   │   │   │           ├── DISFA_train_label_fold3.txt
│   │   │   │           ├── DISFA_weight_fold1.txt
│   │   │   │           ├── DISFA_weight_fold2.txt
│   │   │   │           └── DISFA_weight_fold3.txt
│   │   │   ├── dataset.py
│   │   │   ├── img
│   │   │   │   └── intro.png
│   │   │   ├── LICENSE
│   │   │   ├── model
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   │   ├── ANFL.cpython-311.pyc
│   │   │   │   │   ├── basic_block.cpython-311.pyc
│   │   │   │   │   ├── graph_edge_model.cpython-311.pyc
│   │   │   │   │   ├── graph.cpython-311.pyc
│   │   │   │   │   ├── MEFL.cpython-311.pyc
│   │   │   │   │   ├── resnet.cpython-311.pyc
│   │   │   │   │   └── swin_transformer.cpython-311.pyc
│   │   │   │   ├── ANFL.py
│   │   │   │   ├── basic_block.py
│   │   │   │   ├── graph_edge_model.py
│   │   │   │   ├── graph.py
│   │   │   │   ├── MEFL.py
│   │   │   │   ├── resnet.py
│   │   │   │   └── swin_transformer.py
│   │   │   ├── OpenGraphAU
│   │   │   │   ├── conf
│   │   │   │   │   └── hybrid_config.yaml
│   │   │   │   ├── conf.py
│   │   │   │   ├── dataset.py
│   │   │   │   ├── demo_imgs
│   │   │   │   │   ├── 10025_pred.jpg
│   │   │   │   │   ├── 1014.jpg
│   │   │   │   │   ├── 10259_pred.jpg
│   │   │   │   │   ├── 25329_pred.jpg
│   │   │   │   │   └── 828_pred.jpg
│   │   │   │   ├── demo.py
│   │   │   │   ├── model
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── ANFL.py
│   │   │   │   │   ├── basic_block.py
│   │   │   │   │   ├── graph_edge_model.py
│   │   │   │   │   ├── graph.py
│   │   │   │   │   ├── MEFL.py
│   │   │   │   │   ├── modeling_finetune.py
│   │   │   │   │   ├── modeling_pretrain.py
│   │   │   │   │   ├── resnet.py
│   │   │   │   │   └── swin_transformer.py
│   │   │   │   ├── README.md
│   │   │   │   ├── sam.py
│   │   │   │   ├── test_stage1.py
│   │   │   │   ├── test_stage2.py
│   │   │   │   ├── tool
│   │   │   │   │   ├── calculate_AU_class_weights.py
│   │   │   │   │   └── dataset_process.py
│   │   │   │   ├── train_stage1.py
│   │   │   │   ├── train_stage2.py
│   │   │   │   └── utils.py
│   │   │   ├── README.md
│   │   │   ├── requirements.txt
│   │   │   ├── test.py
│   │   │   ├── tool
│   │   │   │   ├── BP4D_calculate_AU_class_weights.py
│   │   │   │   ├── BP4D_deal_AU_relation.py
│   │   │   │   ├── BP4D_image_label_process.py
│   │   │   │   ├── DISFA_calculate_AU_class_weights.py
│   │   │   │   ├── DISFA_deal_AU_relation.py
│   │   │   │   ├── DISFA_image_label_process.py
│   │   │   │   └── README.md
│   │   │   ├── train_stage1.py
│   │   │   ├── train_stage2.py
│   │   │   ├── utils.py
│   │   │   └── weights
│   │   └── pos.py
│   ├── data
│   │   └── loaders.py
│   ├── deep_mers.py
│   ├── experiments
│   │   └── cfg1.yml
│   ├── metrics
│   ├── model_config.yml
│   ├── pooler
│   │   └── base_pooler.py
│   └── train.py
├── uv.lock
└── worksheet.ipynb

30 directories, 120 files
