# ECCV2024 Dataset Distillation Competition Implementation (Track 2) using Efficient Dataset Distillation via Minimax Diffusion

Reference article of "[Efficient Dataset Distillation via Minimax Diffusion](https://arxiv.org/abs/2311.15529)".




## Getting Started


1. Set up the environment:
```bash
conda create -n diff python=3.8
conda activate diff
pip install -r requirements.txt
```

2. Prepare the pre-trained DiT model:
```bash
python download.py
```

3. Prepare the tiny image net and Cifar 100 and save into ./dataset folder, we shall see two folders: `./dataset/cifar-100-python` and `./dataset/tiny-imagenet-200`. Can download from [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html) and imagenet_tiny:

```python
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```


4. The Test Set of CIFAR-100 and Tiny-ImageNet can be downloaded [reference_data.zip](https://drive.google.com/file/d/1MZMsEbBHe3gYrq4y4Na3Ogh9sIKecng-/view?usp=drive_link). Please unzip the reference testing data "reference_data.zip" and create the folder structure "./reference_data/{cifar100|tinyimagenet}_test.pt"

Please keep in mind, that the test data is normalized following the standard normalization technqiues for CIFAR100 and TinyImagenet. In particular we assume your distilled data has been learned from a normalized training dataset using:

```python
#* CIFAR100
# mean = [0.5071, 0.4866, 0.4409]
# std = [0.2673, 0.2564, 0.2762]

#* TinyImagenet
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

```

5. To download pretrain weights .pt documents, please go to [here](https://entuedu-my.sharepoint.com/:f:/g/personal/fanj0019_e_ntu_edu_sg/Es4pJj-0MbRMg-WotePsQP4B5j2nrZazT1rytXvkqZvRZg?e=3NbbWg). Please save the weights .pt documents  to `./submission` folder.

## Running Commands


### Dataset fine-tuning
For Imagenet Tiny, please run
```bash
torchrun --nnode=1 --master_port=25678 train_dit.py --model DiT-XL/2  --data-path dataset/tiny-imagenet-200/train/ --ckpt pretrained_models/DiT-XL-2-256x256.pt --global-batch-size 8 --tag minimax --ckpt-every 12500 --log-every 1250 --epochs 10   --finetune-ipc -1 --results-dir ./logs/run-0 --spec imagenet_tiny --nclass 200 --condense
```
For Cifar 100, we currently do not implement min-max loss for dataset finetuning, please run：
```bash
torchrun --nnode=1 --master_port=25678 train_dit.py --model DiT-XL/2  --data-path dataset/tiny-imagenet-200/train/ --ckpt pretrained_models/DiT-XL-2-256x256.pt --global-batch-size 8 --tag minimax --ckpt-every 6250 --log-every 1250 --epochs 10     --finetune-ipc -1 --results-dir ./logs/run-1 --spec cifar100 --nclass 100
```

We provide pretrained model weights for imagenet tiny in the following folder: `pretrain_imagenet_cifar/imagenet_tiny.pt` and `pretrain_imagenet_cifar/cifar.pt`.

### Evaluation
In the evaluation, we first take 10 mins to run the data generation, and take 100 epochs to train the Convnet. The testing script follows the ECCV 2024 challenge provided scripts. To download pretrain weights .pt documents, please go to [here](https://entuedu-my.sharepoint.com/:f:/g/personal/fanj0019_e_ntu_edu_sg/Es4pJj-0MbRMg-WotePsQP4B5j2nrZazT1rytXvkqZvRZg). Please save the weights .pt documents  to `./submission` folder.

```bash
python evaluate.py --ckpt ./submission/imagenet_tiny.pt,./submission/cifar.pt
```
```bash
python evaluate.py --ckpt [path_to_imagenet].pt,[path_to_cifar].pt
```
## Important Hyperparameters
For dataset fine-tuning, in train_dit.py args: `--lambda-neg` and `--lambda-pos` are the loss weights for min-max loss. `--lr` is the learning rate. The three terms are important and should be tuned together.
For Evaluation, in evaluate.py args: `--num-sampling-steps` is the diffusion steps, determine the tradeoff between image quality and inference speed.


