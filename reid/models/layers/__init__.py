from torch import nn
from torch.autograd import Variable

import torch
import warnings

from torch.nn import DataParallel as DataParallel_
from collections import OrderedDict

from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel.replicate import _broadcast_coalesced_reshape


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def param_classifier(self):
        for name, param in self.named_params(self):
            if name == 'classifier.weight':
                yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                if p.requires_grad is not False:
                    yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):

                name_t, param_t = tgt
                grad = src

                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:  # ignore classifier's weight which is not used
                    tmp = param_t - lr_inner * grad

                # print(type(tmp))
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        if not isinstance(self.module, MetaModule):
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids)

        params = kwargs.pop('params', None)
        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        # Add params argument unchanged back in kwargs
        replicas = self._replicate_params(params, inputs_, device_ids,
                                          detach=not torch.is_grad_enabled())
        kwargs_ = tuple(dict(params=replica, **kwarg)
                        for (kwarg, replica) in zip(kwargs_, replicas))
        return inputs_, kwargs_

    def _replicate_params(self, params, inputs, device_ids, detach=False):
        if params is None:
            module_params = OrderedDict(self.module.named_parameters())
        else:
            # Temporarily disable the warning if no parameter with key prefix
            # `module` was found. In that case, the original params dictionary
            # is used.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                module_params = self.get_subdict(params, key='module')
            if module_params is None:
                module_params = params

        replicas = _broadcast_coalesced_reshape(list(module_params.values()),
                                                device_ids[:len(inputs)],
                                                detach)
        replicas = tuple(OrderedDict(zip(module_params.keys(), replica))
                         for replica in replicas)
        return replicas
