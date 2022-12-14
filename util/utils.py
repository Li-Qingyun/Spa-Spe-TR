import os
import sys
import random
import logging
import functools
from termcolor import colored
from collections import namedtuple

import numpy as np

import torch
import torchvision.transforms as T
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform


def _merge_kwargs(_kwargs, _default_kwargs):
    for k, v in _kwargs.items():
        _default_kwargs[k] = v
    return _default_kwargs


class false_logger:
    """An expedient."""

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def info(msg):
        print(msg)

    @staticmethod
    def warn(msg):
        print(msg)

    @staticmethod
    def warning(msg):
        print(msg)


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=None):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        name = '_' + name if name is not None else name
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log{name}.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def save_checkpoint(output_dir=None, ckpt_interval=10, epoch=None, model: torch.nn.Module=None,
                    optimizer: torch.optim.Optimizer=None,
                    lr_scheduler=None, logger=None, max_acc=None,
                    val_acc=None, lr_drop=None, save_path=None):
    _none = lambda : None
    logger = false_logger if logger is None else logger
    save_state = {'model': getattr(model, 'state_dict', _none)(),
                  'optimizer': getattr(optimizer, 'state_dict', _none)(),
                  'lr_scheduler': getattr(lr_scheduler, 'state_dict', _none)(),
                  'max_accuracy': max_acc,
                  'epoch': epoch,}
    save_state = {k:v for k, v in save_state.items() if v is not None}

    if save_path is None:
        if epoch is not None:
            if (epoch + 1) % ckpt_interval == 0:
                save_path = os.path.join(output_dir, f'ckpt_epoch_{epoch}.pth')
        if lr_drop is not None:
            if (epoch + 1) == lr_drop:
                save_path = os.path.join(output_dir, f'ckpt_epoch_{epoch}_before_drop.pth')
        if val_acc is not None and max_acc is not None:
            if val_acc > max_acc:
                save_path = os.path.join(output_dir, 'best_ckpt.pth')
    if save_path is not None:
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved.")


def get_dataloaders(datasets, args, test_batch_mult=1, return_dict=False):
    Dataloaders = namedtuple('Datasets', ['train', 'val', 'test'])
    data_loader_train = DataLoader(datasets.train, args.batch_size,
                                   drop_last=False, num_workers=args.num_workers,
                                   shuffle=True, pin_memory=True)
    data_loader_val = DataLoader(datasets.val, int(args.batch_size * test_batch_mult),
                                 drop_last=False, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True)
    data_loader_test = DataLoader(datasets.test, int(args.batch_size * test_batch_mult),
                                  drop_last=False, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)
    loader_dict = {'train': data_loader_train, 'val': data_loader_val, 'test': data_loader_test}
    return loader_dict if return_dict else Dataloaders(**loader_dict)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def resume_ckpt(model, optimizer, lr_scheduler, args):
    checkpoint = torch.load(args.resume)

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)

    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        import copy
        p_groups = copy.deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg, pg_old in zip(optimizer.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']
        print(optimizer.param_groups)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
        args.override_resumed_lr_drop = True
        if args.override_resumed_lr_drop:
            print(
                'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
            lr_scheduler.step_size = args.lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        lr_scheduler.step(lr_scheduler.last_epoch)
        args.start_epoch = checkpoint['epoch'] + 1


def get_flops(
        model: torch.nn.Module,
        trainset: torch.utils.data.Dataset,
        logger = None,
):
    logger = false_logger() if logger is None else logger

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    sample = trainset[0][0].unsqueeze(0).cuda()
    flops = FlopCountAnalysis(model, sample)
    flops.total()
    if isinstance(sample, torch.Tensor): print(sample.shape)
    logger.info(flop_count_table(flops, max_depth=2, activations=None, show_param_shapes=False))


def get_mixup_fn(smooth, num_classes: int, **kwargs):
    default_kwargs = dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=smooth,
        num_classes=num_classes
    )
    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
    return Mixup(**kwargs)


def get_transform(img_size, test_crop: bool = True, split: str = None,
                  imagenet_transform: bool = True, **kwargs):
    """get transform of ImageNet, used in RSSC datasets"""
    if img_size is None:
        return None
    if not imagenet_transform:
        return T.Compose([T.ToTensor(), T.Resize((img_size, img_size))])

    _splits = ('train', 'val', 'test')
    assert split in _splits or split is None, f'split={split}, choose in {_splits}'
    Transforms = namedtuple('Transforms', _splits)

    try:
        from torchvision.transforms.functional import InterpolationMode
        def _interp(method):
            if method == 'bicubic':
                return InterpolationMode.BICUBIC
            elif method == 'lanczos':
                return InterpolationMode.LANCZOS
            elif method == 'hamming':
                return InterpolationMode.HAMMING
            else:
                # default bilinear, do we want to allow nearest?
                return InterpolationMode.BILINEAR
    except:
        from timm.data.transforms import str_to_pil_interp as _interp

    default_kwargs = dict(
        color_jitter=0.4,  # or None,
        auto_augment='rand-m9-mstd0.5-inc1',  # or None,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
    )
    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v

    resize_im = img_size > 32

    # this should always dispatch to transforms_imagenet_train
    train_transform = create_transform(
        input_size=img_size,
        is_training=True,
        **kwargs
    )
    if not resize_im:
        print(f'img_size={img_size}: '
              'Replace RandomResizedCropAndInterpolation with RandomCrop')
        train_transform.transforms[0] = transforms.RandomCrop(img_size, padding=4)

    t = []
    if resize_im:
        if test_crop:
            size = int((256 / 224) * img_size)
            # size = img_size
            t.append(
                transforms.Resize(size, interpolation=_interp(kwargs['interpolation'])),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(img_size))
        else:
            t.append(
                transforms.Resize((img_size, img_size),
                                  interpolation=_interp(kwargs['interpolation']))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    val_transform = transforms.Compose(t)

    test_transform = val_transform

    ts = Transforms(train_transform, val_transform, test_transform)
    return getattr(ts, split) if split is not None else ts

def load_pretrained_swin(pretrained, model, logger):
    logger.info(f"==============> Loading weight {pretrained} for fine-tuning......")
    checkpoint = torch.load(pretrained, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{pretrained}'")

    del checkpoint
    torch.cuda.empty_cache()