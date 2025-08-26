### Installation
```
Requirements:
torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0
ftfy==5.8
regex==2023.10.3 
tqdm==4.65.0
transformers==4.25.0
bytecode ==0.15.1
matplotlib==3.8.0
scikit-learn ==1.3.0
opencv-python==4.9.0.80
pyyaml==6.0.1
clip==0.2.0
timm==0.9.16
tensorboardX==2.6.2.2
easydict==1.13
chardet==5.2.0
```

### Prepare Pre-trained Models
The file tree should be
```
logs
└── pretrained
    └── pass_vit_base_full.pth
    └── ALBEF.pth
bert-base-uncased
└── pytorch_model.bin
```

download pretrained model [pass_vit_base_full.pth](https://drive.google.com/file/d/1sZUrabY6Lke-BJoxOEviX5ALJ017x4Ft/view) and [ALBEF.pth](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth) to pretrained directory

download [pytorch_model.bin](https://huggingface.co/google-bert/bert-base-uncased/blob/main/pytorch_model.bin) to bert-base-uncased

### Prepare data
The file tree should be
```
data
└── ltcc
    └── datalist
        └── query_sc.txt
        └── gallery_sc.txt
        └── query_cc.txt
        └── gallery_cc.txt
        └── query_general.txt
        └── gallery_general.txt
        └── train.txt
    └── LTCC_ReID
└── market
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── Market-1501
└── llcm
    └── LLCM
    └── query.txt
    └── gallery.txt
    └── train.txt
```

### Training

```
shell
./scripts/train.sh transformer_dualattn_joint ${description}
# e.g., sh ./scripts/train.sh transformer_dualattn_joint test
```


### Testing

```
shell
./inference/${dataset}.sh ${description} ${checkpoint}
# e.g., sh ./inference/market.sh test model_best
```