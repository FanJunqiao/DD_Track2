# Main program

# Imports
import sys
import os
os.system('nvidia-smi')


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm
from utils.utils import DiffAugment, ParamDiffAug
from utils.network import ConvNet
from sample import *
from datetime import datetime 
from torchvision import transforms

#!#################################################################################
#!#################################################################################

#* List of Normalizations (Expect testing data to be normalized with):

#* CIFAR100
# mean = [0.5071, 0.4866, 0.4409]
# std = [0.2673, 0.2564, 0.2762]


#* TinyImagenet
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

#!#################################################################################
#!#################################################################################


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



# 
def epoch(args, dataset_idx, mode, dataloader, net, optimizer, criterion, aug, dsa_param, device='cuda'):
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    # if mode == "test" :
    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        # print(img.min(), img.max(), img.shape)
        if aug:
            img = DiffAugment(img, dsa_strategy, param=dsa_param)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def sample(args, dataset_idx, device = 'cuda'):
    # Labels to condition the model
    with open('./misc/wnids.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == "imagenet_tiny":
        file_list = './misc/wnids.txt'
    else:
        file_list = './misc/class100.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass[dataset_idx] * phase
    cls_to = args.nclass[dataset_idx] * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    print(len(sel_classes))
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt[dataset_idx] or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    # checkpoint = {
    #                     "model": state_dict
    #                 }
    # checkpoint_path =  ckpt_path.replace(".pt", "_new.pt")
    # torch.save(checkpoint, checkpoint_path)
    # print(f"Saved checkpoint to {checkpoint_path}")
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    batch_size = 32


    start_time = datetime.now() 
    data_list = []
    label_list = []
    total_time = 10 * 60
    shift = 0
    print("Starting dataset generation: ", start_time)
    os.makedirs(os.path.join(args.save_dir), exist_ok=True)
    with torch.no_grad():
        while (True):
            # os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
            # for shift in tqdm(range(args.num_samples // batch_size)):
            class_label = np.random.randint(args.nclass[dataset_idx], size=batch_size)
            # sel_class = [sel_classes[i] for i in class_label]
            # Create sampling noise:
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor(class_label, device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([0] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample #
            
            if dataset_idx == 0:
                resize = transforms.Resize([64,64])
                renorm1 = transforms.Normalize(mean=[0,0,0], std=[2, 2, 2], inplace=True)
                renorm2 = transforms.Normalize(mean=[-0.5, -0.5, -0.5],std=[1, 1, 1], inplace=True)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
            elif dataset_idx == 1:
                resize = transforms.Resize([32,32])
                renorm1 = transforms.Normalize(mean=[0,0,0], std=[2, 2, 2], inplace=True)
                renorm2 = transforms.Normalize(mean=[-0.5, -0.5, -0.5],std=[1, 1, 1], inplace=True)
                normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], inplace=True)
            img = normalize(renorm2(renorm1(resize(samples))))
            lab = torch.tensor(class_label)
            # print(img.shape, img.min(), img.max())

            data_list.append(img.cpu().data)
            label_list.append(lab.cpu().data)

            
        

            # # Save and display images:
            # for image_index, image in enumerate(samples):
            #     save_image(image, os.path.join(args.save_dir,
            #                                     f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))
            
            shift = shift + 1
            current_time = datetime.now() 
            print("Batch time: ", (current_time - start_time).total_seconds(), " s")
            if (current_time - start_time).total_seconds() > total_time:
                break

        data_all = torch.cat(data_list, dim=0)
        label_all = torch.cat(label_list, dim=0)
        print("Generatred dataset shape: ", data_all.shape, label_all.shape)

        del vae, model
    

    return data_all, label_all

def evaluator(args, eval_run, truth_file_path, gpu_available):

    #! Load the distilled train data:
    #! Assert Data should be unnormalized !

    # if gpu_available:
    device = 'cuda'
    # else:
    #     device = 'cpu'


    if eval_run == "cifar100":
        datashape = 32
        num_classes = 100
        idx = 1
        
        

    if eval_run == "tinyimagenet":
        datashape = 64
        num_classes = 200
        idx = 0
        
    

    testdata = torch.load(truth_file_path, map_location='cpu')
    dst_test = TensorDataset(testdata["images_val"], testdata["labels_val"])

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=True, num_workers=0)

    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    param = ParamDiffAug()

    scores = 0
    rounds = 3
    for _ in range(rounds):

        images_train, labels_train = sample(args, idx)
        images_train = images_train.to(device)
        labels_train = labels_train.to(device)
        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=0)

        net = ConvNet(channel=3, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=(datashape, datashape))
        net = net.to(device)
        lr = float(0.01)
        Epoch = int(100)
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss().to(device)

        for ep in tqdm(range(Epoch+1)):
            _, _ = epoch(args, idx, 'train', trainloader, net, optimizer, criterion, aug = True, dsa_param=param, device=device)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        # _, _ = epoch(args, idx, 'train', trainloader, net, optimizer, criterion, aug = True, dsa_param=param, device=device)
        _, acc_test = epoch(args, idx, 'test', testloader, net, optimizer, criterion, aug = False, dsa_param=None, device=device)
        
        scores += acc_test
        print(acc_test)
    return scores / rounds



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument('--truth_dir', type=str, default="./reference_data/")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=list_of_strings, default=["./logs/run-0/030-DiT-XL-2-minimax/checkpoints/0062500.pt", "./logs/run-1/001-DiT-XL-2-minimax/checkpoints/0062500.pt"],
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='imagenet_tiny', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='./results/dit-distillation/imagenet-10-200-minimax', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=10, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=list, default=[200, 100], help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    

    args = parser.parse_args()


    # Paths
    truth_dir = args.truth_dir


    output_file = open(os.path.join('scores.txt'), 'w')


    score = 0
    perf_list = []
    gpu_available = torch.cuda.is_available()
    for eval_run in [ "tinyimagenet","cifar100", ]:
        truth_file_path = "{}{}_test.pt".format(args.truth_dir, eval_run)
        perf = evaluator(args, eval_run=eval_run, truth_file_path=truth_file_path, gpu_available=gpu_available)
        score += perf
        perf_list.append(perf)
        print("Performance on {} dataset is {}".format(eval_run, perf))
    score /= 2
    print("Average Performance is {}, Tiny_imagenet: {}, Cifar_100: {}".format(score, perf_list[0], perf_list[1]))


    output_file.write("correct: {}".format(score))
    output_file.close()
    print('End of program')



