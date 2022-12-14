import time
import argparse
import datetime
from pathlib import Path
from vit_pytorch import ViT
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from datasets import *
from engine import evaluate, train_one_epoch
from util import (create_logger, save_checkpoint,
                  get_dataloaders, set_seed, resume_ckpt,
                  get_flops, get_mixup_fn, get_transform,
                  build_optimizer, build_scheduler, build_criterion)
from Fusion import branch_fusion

def parse_args():
    parser = argparse.ArgumentParser('Two-Branch Pure Transformer for Hyperspectral Image Classification', add_help=False)
    parser.add_argument('--model', default='swin_tiny',
                        choices=['swin_tiny', 'swin_base'], type=str)
    parser.add_argument('--end_stage', default=3,
                        choices=[1, 2, 3, 4], type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=250, type=int)
    parser.add_argument('--ckpt_interval', default=250, type=int)

    parser.add_argument('--clip_max_norm', default=0, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--seed', default=1, type=int,
                        help='base seed for experiments (seed + 1~10)')

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--scheduler', default='cosine', type=str)
    parser.add_argument('--flops', action='store_true')

    parser.add_argument('--dataset_root', default='G:/Pavia',type=str)
    parser.add_argument('--output_dir', default='logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--exp_name', default='swin_hsi_cls', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # training tricks
    parser.add_argument('--mixup', default=False, type=bool,
                        help='mixup and cutmix data argumentation')
    parser.add_argument('--smooth', default=0., type=float,
                        help='label smoothing')

    args, unparsed = parser.parse_known_args()
    return args


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class Swin_hsi(nn.Module):

    def __init__(self,
                 name: str = 'swin_tiny',
                 patch_size: int = 4,
                 num_classes: int = 0,
                 num_bands: int = 0,
                 end_stage: int = None,
                 keep_patch_embed: bool = True):
        assert num_classes > 0 and num_bands > 0
        super(Swin_hsi, self).__init__()
        self.end_stage = end_stage

        self.input_proj = nn.Conv2d(num_bands, 3, 1)

        from models.swin_backbone import config_dict, SwinTransformer
        config = config_dict[name]
        config['strides'] = (patch_size, 2, 2, 2)
        config['patch_size'] = patch_size

        if end_stage is not None:
            assert isinstance(end_stage, int) and 0 < end_stage <= len(config['depths'])

        self.body = SwinTransformer(**config)
        if keep_patch_embed and patch_size != 4:
            self.body.patch_embed.projection = nn.Conv2d(3, 96, 4, patch_size)

        delete_keys = ['patch_embed.projection.weight'] \
            if patch_size != 4 and not keep_patch_embed else None
        print(self.body.init_weights(delete_keys))
        freeze(self.body.stages[0])

        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        token_dim = self.body.stages[end_stage-1 if end_stage is not None else -1].embed_dims
        self.output_proj = nn.Sequential(nn.Linear(token_dim, num_classes))


    def forward(self, x):
        x = self.input_proj(x)
        xs = self.body(x, self.end_stage)
        token = self.avg_pool(xs[self.end_stage-1]).flatten(1)
        return token

    @torch.no_grad()
    def print_output_sizes(self, sample_size):
        print('----------------------')
        print(f'Input Size : \n\t{sample_size}')
        sample = torch.randn(*sample_size)
        outs = self.body(self.input_proj(sample), self.end_stage)
        print('Output Size :')
        for i, out in enumerate(outs):
            print(f'\t{i}:{out.shape}')
        print('----------------------')

class spectral_spatial(nn.Module):
    def __init__(self,model_spectral,model_spatial, num_classes):
        super(spectral_spatial, self).__init__()
        self.model_spectral=model_spectral
        self.model_spatial=model_spatial
        self.fc=nn.Linear(384+64,num_classes)
    def forward(self, x1, x2):
        x_spectral = self.model_spectral(x2)
        x_spatial =  self.model_spatial(x1)
        x_all = torch.hstack([x_spectral, x_spatial])
        # x_all=x_spectral+x_spatial
        output=self.fc(x_all)
        return output


def main(args, logger, seed, exp_time, acc, A, k):

    set_seed(seed)

    # build dataset and dataloader
    builder = PaviaBuilder(args.dataset_root, seed=seed)
    data_sets = builder.get_datasets()
    data_loaders = get_dataloaders(data_sets, args, 4)
    num_classes = builder.get_num_classes()
    num_bands = builder.get_num_bands()
    logger.info(f'Dataset builder info: \n{builder}')


    mixup_fn = get_mixup_fn(args.smooth, num_classes) if args.mixup else None


    # build model
    model_spatial = Swin_hsi(args.model, 4, num_classes, num_bands, args.end_stage).cuda()

    model_spectral = ViT(
        dim=64,
        image_size=1,
        patch_size=1,
        num_classes=num_classes,
        num_patches=num_bands,
        depth=5,  # 12
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,

    ).cuda()

    logger.info(str(model_spatial))

    model = spectral_spatial(model_spectral, model_spatial, num_classes).cuda()
    model.add_module("Fusion", branch_fusion(384+64))
    print(model)
    # calculate params and flops
    total_parameters = sum(p.numel() for p in model.parameters())
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'total params: {total_parameters} | train params: {n_parameters}')
    if hasattr(model, 'flops'):
        logger.info(f"number of GFLOPs: {model.flops() / 1e9}")
    elif args.flops:
        get_flops(model, data_sets.train, logger)

    param_dicts=[{"params": [p for p in model.model_spatial.parameters() if model_spatial], "lr": 0.0002}, # 0.0002 90.52
                 {"params": [p for p in model.model_spectral.parameters() if model_spectral],
                  "lr": 5e-4}]
    optimizer = optim.Adam(param_dicts)

    logger.info(f'{optimizer.__class__} is loaded as the optimizer.')
    lr_scheduler = build_scheduler(args.scheduler, optimizer, len(data_loaders.train),
                                   epochs=args.epochs, lr=args.lr, decay_epochs=args.lr_drop)
    logger.info(f'{lr_scheduler.__class__} is loaded as the lr_scheduler.')
    criterion = build_criterion(mixup=args.mixup, smooth=args.smooth)
    logger.info(f'{criterion.__class__} is loaded as the criterion.')

    best_acc = 0.0
    # auto_resume, resume and pretrain  # TODO
    if args.resume:
        resume_ckpt(model, optimizer, lr_scheduler, args)

    # ###################### TRAIN_VAL ###################### #
    print("Start training")

    train_start=time.time()
    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            model, data_loaders.train, criterion, optimizer, lr_scheduler, mixup_fn,
            args.clip_max_norm, epoch, args.epochs, logger)

        if epoch % 2 == 1 and (epoch + 1) > (args.epochs - 10):

            val_acc = evaluate(model,data_loaders.test, criterion, acc, A, k,  exp_time, logger=logger)[0]

            save_checkpoint(args.output_dir, args.ckpt_interval, epoch, model, optimizer,
                            lr_scheduler, logger, best_acc, val_acc, args.lr_drop)
            best_acc = max(best_acc, val_acc)
    train_end=time.time()
    print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
    # ###################### Evaluate ###################### #
    test_acc, _ , acc, A, k= evaluate(model,data_loaders.test, criterion, acc, A, k, exp_time, logger=logger)
    save_checkpoint(model=model, save_path=Path(args.output_dir) / f'last_model_{test_acc:.4f}.pth')
    logger.info(f'Last model accuracy: {test_acc:.4f}')

    model.load_state_dict(torch.load(Path(args.output_dir) / 'best_ckpt.pth')['model'])
    test_acc, _, acc, A, k= evaluate(model,data_loaders.test, criterion, acc, A, k,exp_time, logger=logger)
    test_end = time.time()

    print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
    save_checkpoint(model=model, save_path=Path(args.output_dir) / f'best_model_{test_acc:.4f}.pth')
    logger.info(f'Best model accuracy: {test_acc:.4f}')
    # print(f'test spectral accuracy:{test_acc_spectral}')
    return test_acc, acc, A, k


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # prepare directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # logger
    logger = create_logger(output_dir=args.output_dir, name=args.exp_name)
    logger.info(args)
    start_time = time.time()
    acc_list = []  # If the experiment is interrupted, provide the previous results here.
    EXP_TIMES = 1
    acc = np.zeros([EXP_TIMES, 1])
    A = np.zeros([EXP_TIMES, 9])
    k = np.zeros([EXP_TIMES, 1])
    for seed in range(len(acc_list) + 1, EXP_TIMES + 1):
        logger.info(f'EXP {seed}: SEED = {args.seed + seed}')
        test_acc, acc, A, k= main(args, logger, args.seed + seed,  seed, acc, A, k)

    AA = np.mean(A, 1)
    AAMean = np.mean(AA, 0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(acc)
    OAStd = np.std(acc)
    kMean = np.mean(k)
    kStd = np.std(k)




    print("average OA: " + "{:.2f}".format(OAMean) + " ± " + "{:.2f}".format(OAStd))
    print("average AA: " + "{:.2f}".format(100 * AAMean) + " ± " + "{:.2f}".format(100 * AAStd))
    print("average kappa: " + "{:.4f}".format(100 * kMean) + " ± " + "{:.4f}".format(100 * kStd))

    for i in range(9):
        print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
    total_time = time.time() - start_time
    print(f'total time: {total_time}')
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Run time: {}'.format(total_time_str))