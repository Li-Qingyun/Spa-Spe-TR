# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch.nn
from torch import optim as optim
from util.utils import _merge_kwargs


def build_optimizer(name: str, model: torch.nn.Module, lr, **kwargs):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = name.lower()
    if opt_lower == 'sgd':
        default_kwargs = dict(momentum=0.9, weight_decay=0.05,)
        optim_cls = optim.SGD
    elif opt_lower == 'adamw':
        default_kwargs = dict(eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05,)
        optim_cls = optim.AdamW
    elif opt_lower == 'adam':
        default_kwargs = dict()
        optim_cls = optim.Adam
    else:
        raise NotImplementedError(f'The {name} is not support for optimizer.')
    kwargs = _merge_kwargs(kwargs, default_kwargs)
    optimizer = optim_cls(parameters, lr=lr, **kwargs)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
