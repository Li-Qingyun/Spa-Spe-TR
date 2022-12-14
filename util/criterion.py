import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def build_criterion(mixup: bool=False, smooth: float=0.):
    if mixup:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif smooth > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion