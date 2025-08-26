from __future__ import absolute_import

import torch, gc
from contextlib import contextmanager


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        output, target = to_torch(output), to_torch(target)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret


def to_cpu(module:torch.nn.Module):
    if module is None:
        return
    m = module.module if hasattr(module, 'module') else module
    m.to('cpu')

def to_cuda(module:torch.nn.Module):
    if module is None:
        return
    m = module.module if hasattr(module, 'module') else module
    m.to('cuda')

def free_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()

@contextmanager
def on_device(module:torch.nn.Module, device='cuda'):
    """
    모듈을 잠깐 device로 옮겼다가 끝나면 원래대로 복귀
    """
    if module is None:
        yield None
        return    
    m = module.module if hasattr(module, 'module') else module
    prev = next(m.parameters()).device
    if str(prev) != device:
        m.to(device)

    try:
        yield m
    finally:
        if str(prev) != device:
            m.to(prev)
        free_cuda_cache()