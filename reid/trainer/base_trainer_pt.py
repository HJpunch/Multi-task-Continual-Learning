import os.path as osp
import time

import torch.distributed as dist

from reid.evaluation.evaluators import Evaluator
from reid.evaluation.evaluators_t import Evaluator as Evaluator_t2i
from reid.utils.meters import AverageMeter
from reid.utils.serialization import save_checkpoint
from tensorboardX.writer import SummaryWriter

from reid.multi_tasks_utils.multi_task_distributed_utils_pt import multitask_reduce_gradients
from reid.utils.serialization import load_checkpoint, copy_state_dict
from reid.utils.feature_tools import initial_classifier

import copy
import torch.nn as nn

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

class BaseTrainer(object):
    def __init__(self, model, args, this_task_info=None):
        super(BaseTrainer, self).__init__()
        self.this_task_info = this_task_info
        self.model = model
        self.args = args
        self.fp16 = args.fp16
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        local_rank = self.this_task_info.task_rank if self.this_task_info is not None else dist.get_rank()
        if local_rank == 0:
            writer_dir = osp.join(self.args.logs_dir, 'data')
            self.writer = SummaryWriter(log_dir=writer_dir)

    def _logging(self, cur_iter):
        local_rank = self.this_task_info.task_rank if self.this_task_info is not None else dist.get_rank()
        if not (cur_iter % self.args.print_freq == 0 and local_rank == 0):
            return
        print('Iter: [{}/{}]\t'
              'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
              'Data {:.3f} ({:.3f})\t'
              .format(cur_iter, self.args.iters,
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                      self.data_time.val, self.data_time.avg, ))

    def _refresh_information(self, cur_iter, lr):
        if not (cur_iter % self.args.refresh_freq == 0 or cur_iter == 1):
            return
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        local_rank = self.this_task_info.task_rank if self.this_task_info is not None else dist.get_rank()
        if local_rank == 0:
            print("lr = {} \t".format(lr))

    def _tensorboard_writer(self, current_iter, data):
        local_rank = self.this_task_info.task_rank if self.this_task_info is not None else dist.get_rank()
        
        if local_rank == 0:
            for key, value in data.items():
                self.writer.add_scalar(key, value, current_iter)

    def _do_valid(self, test_loader, query, gallery, validate_feat):
        assert query is not None and gallery is not None
        print('=' * 80)
        print("Validating....")
        self.model.eval()

        if 't2i' in self.this_task_info.task_name:
            evaluator = Evaluator_t2i(self.model, validate_feat)
        else:
            evaluator = Evaluator(self.model, validate_feat)
        mAP, top1 = evaluator.evaluate(test_loader, query, gallery)
        self.model.train()
        print('=' * 80)
        return mAP, top1

    def _parse_data(self, inputs):
        imgs, _, pids, _, indices = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def run(self, inputs):
        raise NotImplementedError
    
    def run_continual(self, inputs):
        raise NotImplementedError

    def train_continual(self,
                  all_train_loader,
                  all_init_loader,
                  all_train_set, 
                  optimizer, 
                  lr_scheduler, 
                  test_loader=None, 
                  query=None, 
                  gallery=None,
                  set_index=0,
                  model_last=None,
                  model_long=None
                  ):
        data_loader = all_train_loader[set_index]
        init_loader = all_init_loader[set_index]
        num_classes = all_train_set[set_index].num_classes

        if set_index <= 1:
            add_num = 0
            old_model = None
        else:
            add_num = sum(
                [all_train_set[i].num_classes for i in range(set_index - 1)]
            )

        if set_index > 0:
            old_model = copy.deepcopy(self.model)
            old_model = old_model.cuda()
            old_model.eval()

            add_num = sum([all_train_set[i].num_classes for i in range(set_index)])
            org_classifier_params = self.model.module.classifier.weight.data
            self.model.module.classifier = nn.Linear(self.model.module.num_features, add_num + num_classes, bias=False)
            self.model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)

            org_classifier_params = self.model.module.classifier_f.weight.data
            self.model.module.classifier_f = nn.Linear(self.model.module.num_features, add_num + num_classes, bias=False)
            self.model.module.classifier_f.weight.data[:add_num].copy_(org_classifier_params)

            org_classifier_params = self.model.module.classifier_c.weight.data
            self.model.module.classifier_c = nn.Linear(self.model.module.num_features, add_num + num_classes, bias=False)
            self.model.module.classifier_c.weight.data[:add_num].copy_(org_classifier_params)

            self.model.cuda()

            global_centers, bio_centers, clot_centers = initial_classifier(self.model, init_loader, self.this_task_info)
            self.model.module.classifier.weight.data[add_num:].copy_(global_centers)
            self.model.module.classifier_f.weight.data[add_num:].copy_(bio_centers)
            self.model.module.classifier_c.weight.data[add_num:].copy_(clot_centers)
            self.model.cuda()

        if set_index > 0:
            model_last.eval()
            if set_index > 1:
                model_long.eval()


        self.model.train()

        end = time.time()
        best_mAP, best_top1, best_iter = 0, 0, 0

        for i, inputs in enumerate(data_loader):
            current_iter = i + 1

            self._refresh_information(current_iter, lr=lr_scheduler.get_lr()[0])
            self.data_time.update(time.time() - end)
            # import pdb;pdb.set_trace()
            loss = self.run_continual(inputs, add_num, model_last, model_long)
            if self.this_task_info:
                loss = self.this_task_info.task_weight * loss

            optimizer.zero_grad()
            if self.fp16:   
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            self.batch_time.update(time.time() - end)
            end = time.time()

            self._logging(current_iter)

            if current_iter % (1 * self.args.save_freq) == 0:
                if test_loader is not None:
                    mAP, top1 = self._do_valid(test_loader, query, gallery, self.args.validate_feat)
                    if best_mAP < mAP:
                        best_mAP = mAP
                        best_iter = current_iter
                    if best_top1 < top1:
                        best_top1 = top1

                    end = time.time()


                if best_iter == current_iter:
                    save_checkpoint({'state_dict': self.model.state_dict()},
                                    fpath=osp.join(self.args.logs_dir,
                                                'checkpoints',
                                                f'{set_index}_{self.this_task_info.test_task_type}',
                                                f'checkpoint_{current_iter}.pth.tar'),)

                print('\n * Finished iterations {:3d}. Best iter {:3d}, Best mAP {:4.1%}.\n'
                      .format(current_iter, best_iter, best_mAP))

            lr_scheduler.step()
        
        return self.model, model_last, model_long