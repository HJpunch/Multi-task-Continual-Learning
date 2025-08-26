import torch.distributed as dist
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss, L1Loss

from reid.loss import TripletLoss
from reid.trainer.base_trainer_pt import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter
import torch
import torch.nn.functional as F

from reid.metric_learning.distance import cosine_similarity
from reid.utils import on_device

class GeneralTransformerTrainer_joint(BaseTrainer):
    def __init__(self, model, args, this_task_info=None):
        super(GeneralTransformerTrainer_joint, self).__init__(model, args, this_task_info)

        self.ce_loss = CrossEntropyLoss().cuda()
        self.triplet_loss = TripletLoss(margin=self.args.margin).cuda()
        self.KLDivLoss = KLDivLoss( reduction = "none")
        self.MSE = MSELoss(size_average=None, reduce=None, reduction='mean')
        self.MAE = L1Loss(size_average=None, reduce=None, reduction='mean')

        self.losses_ce = AverageMeter()
        self.losses_bme = AverageMeter()
        self.losses_tr = AverageMeter()
        self.precisions = AverageMeter()
        self.losses_last = AverageMeter()
        self.losses_long = AverageMeter()

    def _logging(self, cur_iter):
        self._tensorboard_writer(cur_iter, data={
            'loss': self.losses_ce.val + self.losses_tr.val,
            'loss_ce': self.losses_ce.val,
            'loss_bme': self.losses_bme.val,
            'loss_tr': self.losses_tr.val,
            'prec': self.precisions.val,
            'losses_last': self.losses_last.val,
            'losses_long': self.losses_long.val
        })
        local_rank = self.this_task_info.task_rank if self.this_task_info else dist.get_rank()
        if not (cur_iter % self.args.print_freq == 0 and local_rank == 0):
            return
        if self.this_task_info:
            task_id, task_name = self.this_task_info.task_id, self.this_task_info.task_name
        else:
            task_id, task_name = 0, 'single task'
        print('Iter: [{}/{}]\t'
              'task{}: {}\t'
              'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss_ce {:.3f} ({:.3f})\t'
              'Loss_tr {:.3f} ({:.3f})\t'
              'Loss_bme {:.3f} ({:.3f})\t'
              'Prec {:.2%} ({:.2%})\t'
              'Loss_last {:.3f} ({:.3f})\t'
              'Loss_long {:.3f} ({:.3f})'
              .format(cur_iter, self.args.iters,
                      str(task_id), str(task_name),
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                      self.data_time.val, self.data_time.avg,
                      self.losses_ce.val, self.losses_ce.avg,
                      self.losses_tr.val, self.losses_tr.avg,
                      self.losses_bme.val, self.losses_bme.avg,
                      self.precisions.val, self.precisions.avg,
                      self.losses_last.val, self.losses_last.avg,
                      self.losses_long.val, self.losses_long.avg,))

    def _refresh_information(self, cur_iter, lr):
        if cur_iter % self.args.refresh_freq == 0 or cur_iter == 1:
            self.batch_time = AverageMeter()
            self.data_time = AverageMeter()
            self.losses_ce = AverageMeter()
            self.losses_tr = AverageMeter()
            self.losses_bme = AverageMeter()
            self.precisions = AverageMeter()
            self.losses_last = AverageMeter()
            self.losses_long = AverageMeter()
            local_rank = self.this_task_info.task_rank if self.this_task_info else dist.get_rank()
            if local_rank == 0:
                print("lr = {} \t".format(lr))

    def _parse_data(self, inputs):
        imgs, instructions, _, _, pids, view_ids, cam_ids, indices = inputs
        
        inputs = imgs.cuda()
        targets = pids.cuda()
        cam_ids = cam_ids.cuda()
        view_ids = view_ids.cuda()
        return inputs, instructions, targets, cam_ids, view_ids

    def run_continual(self, inputs, add_num, model_last=None, model_long=None):
        if 't2i' in self.this_task_info.task_name:
            inputs, instructions, targets, cam_ids, view_ids = self._parse_data(inputs)
            img_feat, text_feat, vl_f, vl_f_n, vl_output, vl_labels, loss_cl, loss_pitm, loss_mlm, loss_mrtd = self.model(inputs, instructions, this_task_info=self.this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)
            
            loss_ce_biometric = loss_mlm + loss_mrtd
            loss_ce_vl = loss_pitm
            
            loss = 0.5*loss_cl + loss_ce_vl + loss_mlm + 0.5*loss_mrtd
            
            self.losses_ce.update(loss_ce_vl.item())
            self.losses_tr.update(loss_cl.item())
            self.losses_bme.update(loss_ce_biometric.item())
            prec, = accuracy(vl_output.data, vl_labels.data)
            
            prec = prec[0]
            self.precisions.update(prec)
        else:
            inputs, instructions, targets, cam_ids, view_ids = self._parse_data(inputs)
            targets += add_num

            feat, bio_f, clot_f, logits1, logits2, logits3, clot_feats_s = self.model(inputs, instructions, this_task_info=self.this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)
            if self.args.fusion_loss=='all':
                if isinstance(logits1, list):
                    ID_LOSS = [self.ce_loss(scor, targets) for scor in logits1[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    loss_ce_biometric = 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits1[0], targets)
                else:
                    loss_ce_biometric = self.ce_loss(logits1, targets)
            if isinstance(feat, list):
                TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                loss_tr_biometric = 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(feat[0], targets, clot_feats_s)[0]
            else:
                loss_tr_biometric = self.triplet_loss(feat, targets, clot_feats_s)[0]
            loss_ce = 0
            loss_tr = 0
            if 'bio' in self.args.fusion_branch:
                if isinstance(logits2, list):
                    ID_LOSS = [self.ce_loss(scor, targets) for scor in logits2[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    loss_ce += 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits2[0], targets)
                else:
                    loss_ce+=self.ce_loss(logits2, targets)
                if isinstance(bio_f, list):
                    TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in bio_f[1:]]
                    TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    loss_tr += 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(bio_f[0], targets, clot_feats_s)[0]
                else:
                    loss_tr+=self.triplet_loss(bio_f, targets, clot_feats_s)[0]
                    
            if 'clot' in self.args.fusion_branch:
                if isinstance(logits3, list):
                    ID_LOSS = [self.ce_loss(scor, targets) for scor in logits3[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    loss_ce += 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits3[0], targets)
                else:
                    loss_ce+=self.ce_loss(logits3, targets)
                if isinstance(clot_f, list):
                    TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in clot_f[1:]]
                    TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    loss_tr += 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(clot_f[0], targets, clot_feats_s)[0]
                else:
                    loss_tr+=self.triplet_loss(clot_f, targets, clot_feats_s)[0]
            
            af_loss = 0
            af_items = 0
            tau = 0.1
            if model_last is not None:
                with torch.no_grad():
                    feat_old, *_ = model_last(inputs, instructions, this_task_info=self.this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)
                Affinity_matrix_new = self.get_normal_affinity(feat, tau)
                Affinity_matrix_old = self.get_normal_affinity(feat_old, tau)
                Affinity_matrix_old_short=Affinity_matrix_old

                divergence, weight, Target_2 = self.cal_KL_old_only(Affinity_matrix_new, Affinity_matrix_old, targets)
                af_loss += divergence
                af_items += 1
                self.losses_last.update(divergence.item())

            if model_long is not None:
                with torch.no_grad():
                    feat_old, *_ = model_long(inputs, instructions, this_task_info=self.this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)
                Affinity_matrix_new = self.get_normal_affinity(feat, tau)
                Affinity_matrix_old = self.get_normal_affinity(feat_old, tau)
                Affinity_matrix_old_long = Affinity_matrix_old

                divergence, weight, Target_1 = self.cal_KL_old_only(Affinity_matrix_new, Affinity_matrix_old, targets)
                af_loss += divergence
                af_items += 1
                self.losses_long.update(divergence.item())

            if af_items > 1:
                Target_1 = (Target_1+Target_2) / 2
                Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
                divergence = self.KLDivLoss(Affinity_matrix_new_log, Target_1)  # 128*128
                divergence = divergence.sum() / Affinity_matrix_new.size(0)
                af_loss = divergence * af_items

                divergence, _, _ = self.dual_cal_KL_old_only(Affinity_matrix_new, Affinity_matrix_old_short,Affinity_matrix_old_long ,targets)
                af_loss = divergence * af_items

            AF_weight = 1.0
            af_loss = af_loss / (af_items+1e-6) * AF_weight



            ratio_rate = 1.0
            if self.args.fusion_loss=='all':
                if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                    loss = ratio_rate * loss_ce / 2 + ratio_rate * self.args.alpha * loss_tr / 2 + loss_ce_biometric + self.args.alpha * loss_tr_biometric
                    loss += af_loss
                else:
                    loss = loss_ce + self.args.alpha * loss_tr + loss_ce_biometric + loss_tr_biometric
            else:
                if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                    loss = loss_ce / 2 + self.args.alpha * loss_tr / 2  
                else:
                    loss = loss_ce + self.args.alpha * loss_tr 

            self.losses_ce.update(loss_ce.item())
            self.losses_tr.update(loss_tr.item())
            if self.args.fusion_loss=='all':
                self.losses_bme.update(loss_ce_biometric.item())
            if 'bio' in self.args.fusion_branch:
                if isinstance(logits2, list):
                    prec, = accuracy(logits2[0].data, targets.data)
                else:
                    prec, = accuracy(logits2.data, targets.data)
            else:
                if isinstance(logits3, list):
                    prec, = accuracy(logits3[0].data, targets.data)
                else:
                    prec, = accuracy(logits3.data, targets.data)
            prec = prec[0]
            self.precisions.update(prec)
        return loss
    
    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    
    def cal_KL_old_only(self,Affinity_matrix_new, Affinity_matrix_old,targets, Gts=None,):
        if Gts == None:
            Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
            Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        # attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
        attri_old = self.get_attri(Gts, Affinity_matrix_old, margin=0)

        '''# prediction is correct on old model'''
        Old_Keep = attri_old['TN'] + attri_old['TP']
        # if torch.any(Old_Keep<1):
        #     print(Old_Keep)
        Target_1 = Affinity_matrix_old * Old_Keep
        # '''# prediction is false on old model but correct on mew model'''
        # New_keep = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])
        # Target_2 = Affinity_matrix_new * New_keep
        '''# missed correct person'''
        Hard_pos = attri_old['FN']
        Thres_P = attri_old['Thres_P']
        Target_3 = Hard_pos * Thres_P

        '''# false wrong person'''
        Hard_neg = attri_old['FP']
        Thres_N = attri_old['Thres_N']
        Target_4 = Hard_neg * Thres_N

        Target__ = Target_1 +  Target_3 + Target_4
        Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization

       
      
        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)  # 128*128
        
        if self.args.weighted_loss:
            Affinity_matrix_new=Affinity_matrix_new/(Affinity_matrix_new.max(-1, keepdim=True)[0]+1e-6)
            weight=torch.abs(Gts-Affinity_matrix_new)
            
            divergence=divergence*weight
            weight=weight.mean(-1)
        else:
            weight=torch.ones(divergence.shape[0]).to(divergence.device)
        divergence=divergence.sum()/Affinity_matrix_new.size(0)

        return divergence,weight, Target
    def dual_cal_KL_old_only(self,Affinity_matrix_new, Affinity_matrix_old_short,Affinity_matrix_old_long ,targets, Gts=None):
        if Gts == None:
            Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
            Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        attri_old_short = self.get_attri(Gts, Affinity_matrix_old_short, margin=0)

        attri_old_long = self.get_attri(Gts, Affinity_matrix_old_long, margin=0)

        dual_keep=(attri_old_short['TN'] + attri_old_short['TP'])*(attri_old_long['TN'] + attri_old_long['TP'])

        Target_1=(Affinity_matrix_old_short*dual_keep+Affinity_matrix_old_long*dual_keep)/2

        single_keep_short=(attri_old_short['TN'] + attri_old_short['TP'])-(attri_old_long['TN'] + attri_old_long['TP'])   # 单个正确,取异或
        single_keep_short=single_keep_short.clamp(min=0.0)  # 取正数值
        Target_2=Affinity_matrix_old_short*single_keep_short

        single_keep_long=(attri_old_long['TN'] + attri_old_long['TP']) -(attri_old_short['TN'] + attri_old_short['TP'])  # 单个正确,取异或
        single_keep_long=single_keep_long.clamp(min=0.0)  # 取正数值
        Target_3=Affinity_matrix_old_long*single_keep_long


        '''# both missed correct person'''
        Hard_pos = attri_old_short['FN'] * attri_old_long['FN']
        Thres_P = torch.maximum(attri_old_short['Thres_P'], attri_old_long['Thres_P'])
        Target_4 = Hard_pos * Thres_P

        '''# both false wrong person'''
        Hard_neg = attri_old_short['FP'] * attri_old_long['FP']
        Thres_N = torch.minimum(attri_old_short['Thres_N'], attri_old_long['Thres_N'])
        Target_5 = Hard_neg * Thres_N


        Target__ = Target_1 +Target_2+  Target_3 + Target_4+Target_5
        Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization

       

        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)  # 128*128


        if self.args.weighted_loss:
            Affinity_matrix_new=Affinity_matrix_new/(Affinity_matrix_new.max(-1, keepdim=True)[0]+1e-6)
            weight=torch.abs(Gts-Affinity_matrix_new)
            
            divergence=divergence*weight
            weight=weight.mean(-1)
        else:
            weight=torch.ones(divergence.shape[0]).to(divergence.device)

        divergence=divergence.sum()/Affinity_matrix_new.size(0)

        if self.args.mse:
            divergence=self.MSE(Target, Affinity_matrix_new)*3000
        elif self.args.mae:
            divergence=self.MAE(Target,Affinity_matrix_new)
        elif self.args.js:
            Target_log=torch.log(Target)
            divergence1=self.KLDivLoss(Target_log, Affinity_matrix_new)  # 128*128
            divergence1=divergence1.sum()/Affinity_matrix_new.size(0)
            divergence=(divergence1+divergence)/2        
        else:
            pass


        return divergence,weight, Target
    
    def get_attri(self, Gts, pre_affinity_matrix,margin=0):
        Thres_P=((1-Gts)*pre_affinity_matrix).max(dim=1,keepdim=True)[0]
        T_scores=pre_affinity_matrix*Gts

        TP=((T_scores-Thres_P)>margin).float()
        try:
            TP=torch.maximum(TP, torch.eye(TP.size(0)).to(TP.device))
        except:
            pass

        FN=Gts-TP
        
        Mapped_affinity=(1-Gts) +pre_affinity_matrix
        try:
            Mapped_affinity = Mapped_affinity+torch.eye(Mapped_affinity.size(0)).to(Mapped_affinity.device)
        except:
            pass
        Thres_N = Mapped_affinity.min(dim=1, keepdim=True)[0]
        N_scores=pre_affinity_matrix*(1-Gts)

        FP=(N_scores>Thres_N ).float()
        TN=(1-Gts) -FP
        attris={
            'TP':TP,
            'FN':FN,
            'FP':FP,
            'TN':TN,
            "Thres_P":Thres_P,
            "Thres_N":Thres_N
        }
        return attris