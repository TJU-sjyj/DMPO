# DMPO



## 1. Environment Setup

```bash
conda config --add channels conda-forge
conda create -f dmpo.yaml
conda activate dmpo
```

## 2. Dataset

### Required Datasets

- **VTAB**: [Download](https://box.nju.edu.cn/f/57d5913e680243fca32b/?dl=1)
- **FGVC Dataset**
    - **CUB-200-2011**: [Download](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
    - **NABirds**: [Download](https://dl.allaboutbirds.org/nabirds)

### Pre-trained Models

- **ViT**: [Download](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)

## 3. Run Script

**Note**: Before running the script, you need to modify the path to your model and dataset.

```bash
bash configs/LoRA/CIFAR100.sh
```

---

> This documentation will be further improved in the future.