from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import random
import sys
import yaml
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import copy
import torch
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
import itertools
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from reid.utils.adamw import AdamW
import torch.nn as nn
import torch.nn.functional as F

from reid import models
from reid.datasets import dataset_entry
from reid.datasets.data_builder_cc import DataBuilder_cc
from reid.datasets.data_builder_sc_mnt import DataBuilder_sc
from reid.datasets.data_builder_t2i import DataBuilder_t2i
from reid.datasets.data_builder_cross import DataBuilder_cross
from reid.trainer import TrainerFactory
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineLR
from reid.utils.distributed_utils_pt import dist_init, dist_init_singletask

from reid.multi_tasks_utils.task_info_pt import get_taskinfo
from reid.multi_tasks_utils.multi_task_distributed_utils_pt import Multitask_DistModule
from reid.utils.serialization import load_checkpoint, copy_state_dict
from reid.utils.feature_tools import extract_features_proto, extract_features_voro

from easydict import EasyDict
from reid.models.layers import DataParallel
from reid.utils.serialization import save_checkpoint
from reid.evaluation.fast_test import fast_eval
from reid.utils.meters import ContinualResults

def configuration():
    parser = argparse.ArgumentParser(description="train simple person re-identification models")

    # distributed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=str, metavar='PATH', default='23446')

    # data
    parser.add_argument('--config', default='scripts/config.yaml')
    parser.add_argument('--data-config', type=str, default=None)
    parser.add_argument('--train-list', type=str, required=True)
    parser.add_argument('--validate', action='store_true', help='validation when training')
    parser.add_argument('--validate_feat', type=str, default='fusion', choices = ['person', 'clothes','fusion'])
    parser.add_argument('--query-list', type=str, default='')
    parser.add_argument('--gallery-list', type=str, default='')
    parser.add_argument('--gallery-list-add', type=str, default=None)
    parser.add_argument('--root', type=str)
    parser.add_argument('--root_additional', type=str, default=None)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--width_clo', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=1)

    # model
    parser.add_argument('--test_feat_type', type=str, choices=['f','f_c','f_b','b','c'])
    parser.add_argument('-t', '--test_task_type', type=str, default='sc', choices=['cc','sc','ctcc','attr', 't2i', 'cross'])
    parser.add_argument('--dropout_clo', type=float, default=0)
    parser.add_argument('--patch_size_clo', type=int, default=16)
    parser.add_argument('--stride_size_clo', type=int, default=16)
    parser.add_argument('--patch_size_bio', type=int, default=16)
    parser.add_argument('--stride_size_bio', type=int, default=16)
    parser.add_argument('--attn_type', type=str, choices=['mix','dual_attn', 'fc'])
    parser.add_argument('--fusion_loss',type=str)
    parser.add_argument('--fusion_branch', type=str)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--vit_type',type=str)
    parser.add_argument('--vit_fusion_layer',type=int)
    parser.add_argument('--pool_clo', action='store_true')
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--aug', action='store_true',help='whether to add auto augmentor')
    parser.add_argument('--colorjitter', type=str, default='all',help='whether to use colorjitter')
    parser.add_argument('--num_features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--metric', type=str, default='linear')
    parser.add_argument('--scale', type=float, default=30.0)
    parser.add_argument('--metric_margin', type=float, default=0.30)

    # optimizer
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='step_lr', choices=['step_lr', 'cosine_lr'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--aug_lr', type=float, default=0.001, help="learning rate of augmentor parameters")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=1000)
    parser.add_argument('--milestones', nargs='+', type=int, default=[7000, 14000],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--aug_start_iter',type=int,default=0)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--iters', type=int, default=24000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=1000)
    parser.add_argument('--refresh-freq', type=int, default=1000)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    parser.add_argument('--fp16', action='store_true', help="training only")
    parser.add_argument('--loss', type=str, default='ce+tri+bio', help='loss function')
    parser.add_argument('--transe_loss', action='store_true', help="whether to use loss of translation relationship")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pretrained', type=str, default='logs/pretrained/pass_vit_base_full.pth', metavar='PATH')

    # continual learning
    parser.add_argument('--weighted_loss', action='store_true', help="the new and old prototypes are used for global information!")
    parser.add_argument('--global_alpha',  type=float, default=400,  help="")  
    parser.add_argument('--absolute_delta', action='store_true', help="only use dual teacher")

    parser.add_argument('--mse', action='store_true', help="using mse loss instead of KL loss")
    parser.add_argument('--mae', action='store_true', help="using mae loss instead of KL loss")
    parser.add_argument('--js', action='store_true', help="using js-diverigence loss instead of KL loss")

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if 'common' in config:
        for k, v in config['common'].items():
            print(k, v)
            setattr(args, k, v)
    args.config = config
    return args


class Runner(object):
    def __init__(self, args):
        super(Runner, self).__init__()
        if args.data_config is not None:
            with open(args.data_config) as f:
                data_config = yaml.load(f, Loader=yaml.FullLoader)
                # data_config = yaml.load(f)
            args.data_config = data_config
        else:
            args.data_config = None
        self.args = args
        self.result = ContinualResults()

    @staticmethod
    def build_optimizer(model, args, lr=None):
        def build_param_groups(model, base_lr, wd):
            no_decay_keys = ("bias", "layernorm", "ln.", "bn.", "batchnorm", "norm.weight", "norm.bias")
            decay, no_decay = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if any(k in n.lower() for k in no_decay_keys) else decay).append(p)
            groups = []
            if decay:    groups.append({"params": decay,    "lr": base_lr, "weight_decay": wd})
            if no_decay: groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0})
            return groups
        
        # params = []
        if lr is None:
            lr = args.lr
        params = build_param_groups(model, lr, args.weight_decay)

        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=lr)
        else:
            raise AttributeError('Not support such optimizer `{}`'.format(args.optimizer))

        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        return model, optimizer

    @staticmethod
    def distributed(model, is_distribuited):
        if is_distribuited:
            model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                            find_unused_parameters=True,broadcast_buffers=False)
        return model

    @staticmethod
    def build_scheduler(optimizer, args):
        if args.scheduler == 'step_lr':
            lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01,
                                             warmup_iters=args.warmup_step+args.aug_start_iter)
        elif args.scheduler == 'cosine_lr':
            lr_scheduler = WarmupCosineLR(optimizer, max_iters=args.iters, warmup_factor=0.01,
                                          warmup_iters=args.warmup_step+args.aug_start_iter)
        else:
            raise AttributeError('Not support such scheduler `{}`'.format(args.scheduler))

        return lr_scheduler

    @staticmethod
    def build_trainer(model_dicts, args, this_task_info=None):
        trainer_factory = TrainerFactory()
        if len(model_dicts.keys()) ==1:
            model = model_dicts['extractor']
            trainer = trainer_factory.create(args.arch, model, args, this_task_info=this_task_info)
        else:
            model = model_dicts['extractor']
            model_aug_bio = model_dicts['aug_bio']
            model_aug_clo = model_dicts['aug_clo']
            trainer = trainer_factory.create(args.arch, model, model_aug_bio, model_aug_clo, args, this_task_info=this_task_info)
        return trainer

    @staticmethod
    def build_validator(args, this_task_info=None):
        if not args.validate:
            return None, None, None
        
        args.query_list = this_task_info.query_list
        args.gallery_list = this_task_info.gallery_list
        print(f"query for: {args.query_list}")

        if this_task_info.task_name == 'DataBuilder_cc':
            print(f"test loader for task: {this_task_info.task_name}")
            data_builder = DataBuilder_cc(args, this_task_info)
        elif this_task_info.task_name == 'DataBuilder_sc':
            print(f"test loader for task: {this_task_info.task_name}")
            data_builder = DataBuilder_sc(args, this_task_info)
        elif this_task_info.task_name == 'DataBuilder_t2i':
            print(f"test loader for task: {this_task_info.task_name}")
            data_builder = DataBuilder_t2i(args, this_task_info)
        elif this_task_info.task_name == 'DataBuilder_cross':
            print(f"test loader for task: {this_task_info.task_name}")
            data_builder = DataBuilder_cross(args, this_task_info)
        else:
            AttributeError('Not support such test type `{}`'.format(this_task_info.task_name))
        test_loader, query_dataset, gallery_dataset = data_builder.build_data(is_train=False)
        return test_loader, query_dataset.data, gallery_dataset.data

    def run(self):
        args = self.args
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))

        if args.data_config is not None:
            tasks = args.data_config['tasks']
            all_task_info = []
            loss_weight_sum = float(np.sum(np.array([task['loss_weight'] for task in tasks.values()])))

            for i, task in enumerate(tasks.values()):
                this_task_info = EasyDict()
                this_task_info.task_id = i
                this_task_info.task_name = task['task_name']
                this_task_info.task_weight = float(tasks[this_task_info.task_id]['loss_weight']) / loss_weight_sum
                this_task_info.train_file_path = task.get('train_file_path', '')
                this_task_info.root_path = task.get('root_path', '')
                this_task_info.task_spec = task.get('task_spec', '')
                this_task_info.attt_file = task.get('attt_file', '')
                this_task_info.query_list = task.get('query_list', '')
                this_task_info.gallery_list = task.get('gallery_list', '')
                this_task_info.test_task_type = task.get('test_task_type', '')
                this_task_info.dataset_name = task.get('dataset_name', '')

                this_task_info.task_root_rank = 0 
                this_task_info.task_rank = 0
                this_task_info.task_size = 1

                all_task_info.append(this_task_info)

        data_builders = [dataset_entry(this_task_info)(args, this_task_info) for this_task_info in all_task_info]
        all_train_loader = []
        all_init_loader = []
        all_train_set = []
        for train_loader, init_loader, train_set in [data_builder.build_data(is_train=True) for data_builder in data_builders]:
            all_train_loader.append(train_loader)
            all_init_loader.append(init_loader)
            all_train_set.append(train_set)

        model = models.create('PASS_Transformer_DualAttn_joint', 
                              num_classes=all_train_set[0].num_classes, 
                              net_config=args, 
                              this_task_info=all_task_info[0])

        checkpoint_p = 'logs/pretrained/ALBEF.pth'
        if checkpoint_p:
            checkpoint = torch.load(checkpoint_p, map_location='cpu')
            state_dict = checkpoint['model']
            state_dict_new = {}
            for key in state_dict.keys():
                if "visual_encoder" not in key:
                    state_dict_new[key] = state_dict[key]
                elif "visual_encoder_m" in key:
                    state_dict_new[key] = state_dict[key]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
            state_dict_new['visual_encoder_m.pos_embed'] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict_new, strict=False)
        model_path = args.pretrained
        if args.vit_type == 'base':
            model.visual_encoder.load_param(model_path,hw_ratio=2)
        print(msg)

        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            copy_state_dict(checkpoint['state_dict'], model, strip='module.')
            print("Resume from checkpoint: {}".format(args.resume))
        
        model.cuda()

        model_dicts = {}
        model, optimizer = self.build_optimizer(model, args)

        model = DataParallel(model)

        lr_scheduler = self.build_scheduler(optimizer, args)
                
        model_dicts['extractor'] = model

        #######################################################################
        trainer = self.build_trainer(model_dicts, args, this_task_info=all_task_info[0])

        for set_index in range(0, len(all_task_info)):
            this_task_info = all_task_info[set_index]
            print(f"train for {this_task_info.task_name}")

            model.module.net_config.test_task_type = this_task_info.test_task_type
            trainer.this_task_info = this_task_info
            
            test_loader, query, gallery = self.build_validator(args, this_task_info)

            print(f"this_task_info - test_task_type: {this_task_info.test_task_type}")
            print(f"model - test_task_type: {model.module.net_config.test_task_type}")
            
            model, *_ = trainer.train_continual(all_train_loader,
                                                all_init_loader,
                                                all_train_set, 
                                                optimizer, 
                                                lr_scheduler, 
                                                test_loader, 
                                                query, 
                                                gallery,
                                                set_index,
                                                )
            
            save_checkpoint({'state_dict': model.state_dict()},
                            fpath=osp.join(args.logs_dir,
                                           'checkpoints',
                                           'model_{}.pth.tar'.format(set_index)),
                                           )

            # forgetting result 저장
            for task in all_task_info[:set_index+1]:
                test_loader, query, gallery = self.build_validator(args, task)
                map, top1 = trainer._do_valid(test_loader, query, gallery, args.validate_feat)
                self.result.add(dataset=task.dataset_name, round_idx=set_index+1, mAP=map, top1=top1)

        self.result.save_json(osp.join(args.logs_dir, 'forgetting_result.json'))
        self.result.plot(title='Forgetting Graph', out_path=osp.join(args.logs_dir, 'forgetting_graph.png'))


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        return new_pos_embed    
    else:
        return pos_embed_checkpoint

def get_num_layer_for_vit(var_name, config):
    if (var_name == "base" or var_name.endswith("prompt_embed_kv")) and config.get('lpe_lr', False):
        return config['num_layers'] - 1
    if var_name in ("base", "base.cls_token", "base.mask_token"):
        return 0
    elif var_name.startswith("base.patch_embed"):
        return 0
    elif var_name.startswith("base") and not (var_name.startswith("base.norm") or
                                                                var_name.startswith("base.ln_pre")):
        if len(var_name.split('.')) < 3:
            import pdb;pdb.set_trace()
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return config['num_layers']  - 1

def count_parameters_num(model):
    count = 0
    count_fc = 0
    param_dict = {name:param for name,param in model.named_parameters()}
    param_keys = param_dict.keys()
    for m_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count += temp_params.data.nelement()
        elif isinstance(m, nn.Linear):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count_fc += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count_fc += temp_params.data.nelement()
    print('Number of conv/bn params: %.2fM' % (count / 1e6))
    print('Number of linear params: %.2fM' % (count_fc / 1e6))
    return count / 1e6, count_fc / 1e6


class AdamWWithClipDev(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_norm=None, norm_type=2):
        import pdb;pdb.set_trace()
        for param in params:
            import pdb;pdb.set_trace()
            if not isinstance(param['params'], torch.Tensor):
                import pdb;pdb.set_trace()
        super(AdamWWithClipDev, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        import pdb;pdb.set_trace()
        self.clip_norm = clip_norm
        self.norm_type = norm_type

        self._split_param_groups = None
        self.reset_split_param_groups()

    def reset_split_param_groups(self):
        if self.clip_norm is not None:
            backbone_param = []
            for x in self.param_groups:
                if x["params"][0].backbone_specific:
                    backbone_param.append(x["params"])
            self._split_param_groups = [_g for _g in [backbone_param] if len(_g) > 0]
            print(f">>> reset_split_param_groups, backbone_param: {len(backbone_param)}")

    def step(self, closure=None):
        if self.clip_norm is not None:
            for _g in self._split_param_groups:
                all_params = itertools.chain(*_g)
                clip_grad_norm_(all_params, self.clip_norm, self.norm_type)

        super(AdamWWithClipDev, self).step(closure)

if __name__ == '__main__':
    cfg = configuration()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    runner = Runner(cfg)
    runner.run()
