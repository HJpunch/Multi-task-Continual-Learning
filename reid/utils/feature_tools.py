import torch
import torch.nn.functional as F

import collections

from reid.utils import on_device

def extract_features(model, data_loader, this_task_info):
    features_all = []
    bio_all = []
    clot_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(data_loader):
            imgs, instructions, fnames, _, pids, view_ids, cam_ids, indices = inputs
            inputs = imgs.cuda()
            targets = pids.cuda()
            cam_ids = cam_ids.cuda()
            view_ids = view_ids.cuda()
            
            features, bio_clot_features, *_ = model(inputs, instructions, this_task_info=this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)
            bio_features, clot_features = bio_clot_features

            for fname, feature, bio_f, clot_f, pid, cid in zip(fnames, features, bio_features, clot_features, pids, cam_ids):
                features_all.append(feature.detach().cpu())
                bio_all.append(bio_f.detach().cpu())
                clot_all.append(clot_f.detach().cpu())
                labels_all.append(int(pid.detach().cpu()))
                fnames_all.append(fname)
                camids_all.append(cid.detach().cpu())
    model.train()
    return features_all, bio_all, clot_all, labels_all, fnames_all, camids_all


def initial_classifier(model, data_loader, this_task_info):
    features_all, bio_all, clot_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader, this_task_info)

    pid2features = collections.defaultdict(list)
    for feature, pid in zip(features_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    global_centers = F.normalize(class_centers, dim=1).float().cuda()

    pid2features = collections.defaultdict(list)
    for feature, pid in zip(bio_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    bio_centers = F.normalize(class_centers, dim=1).float().cuda()

    pid2features = collections.defaultdict(list)
    for feature, pid in zip(clot_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    clot_centers = F.normalize(class_centers, dim=1).float().cuda()

    return global_centers, bio_centers, clot_centers


def extract_features_proto(model, data_loader, this_task_info, get_mean_feature=True):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(data_loader):
            imgs, instructions, fnames, _, pids, view_ids, cam_ids, indices = inputs
            inputs = imgs.cuda()
            targets = pids.cuda()
            cam_ids = cam_ids.cuda()
            view_ids = view_ids.cuda()
            features, *_ = model(inputs, instructions, this_task_info=this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)

            for fname, feature, pid, cid in zip(fnames, features, pids, cam_ids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
   
    if get_mean_feature:
        features_collect = {}
        
        for feature, label in zip(features_all, labels_all):
            if label in features_collect:
                features_collect[label].append(feature)                
            else:
                features_collect[label] = [feature]
                
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]
        vars_mean=[]
        for x in labels_named:
            if x in features_collect.keys():
                feats=torch.stack(features_collect[x])
                feat_mean=feats.mean(dim=0)
                features_mean.append(feat_mean)

                vars_mean.append(feats.std(0))
                # vars_mean.append(torch.sqrt(vars_2))
           
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named,torch.stack(vars_mean)
    else:
        return features_all, labels_all, fnames_all, camids_all


def extract_features_voro(model, data_loader, this_task_info, get_mean_feature=False):    

    features_all, _, _, labels_all, fnames_all, camids_all = extract_features(model, data_loader, this_task_info)

    if get_mean_feature:
        features_collect = {}        

        for feature, label in zip(features_all, labels_all):
            if label in features_collect:
                features_collect[label].append(feature)               
            else:
                features_collect[label] = [feature]                
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]        
        for x in labels_named:
            if x in features_collect.keys():
                features_mean.append(torch.stack(features_collect[x]).mean(dim=0))                
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named
    else:
        return features_all, labels_all, fnames_all, camids_all
    

def extract_features_uncertain(model, data_loader, this_task_info, get_mean_feature=False):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    var_all=[]
    model.train()
    
    with torch.no_grad():
        for i, inputs in enumerate(data_loader):
            imgs, instructions, fnames, _, pids, view_ids, cam_ids, indices = inputs
            inputs = imgs.cuda()
            targets = pids.cuda()
            cam_ids = cam_ids.cuda()
            view_ids = view_ids.cuda()
            # TODO: train 모드에서 모델이 out_var을 반환하도록 수정.
            features, *_, out_var = model(inputs, instructions, this_task_info=this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids, dkp=True)

            for fname, feature, pid, cid, var in zip(fnames, features, pids, cam_ids, out_var):
                features_all.append(feature.detach().cpu())
                labels_all.append(int(pid.detach().cpu()))
                fnames_all.append(fname)
                camids_all.append(cid.detach().cpu())
                var_all.append(var.detach().cpu())

    if get_mean_feature:
        features_collect = {}
        var_collect = {}

        for feature, label, var in zip(features_all, labels_all,var_all):
            if label in features_collect:
                features_collect[label].append(feature)
                var_collect[label].append(var)
            else:
                features_collect[label] = [feature]
                var_collect[label]=[var]
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]
        vars_mean=[]
        for x in labels_named:
            if x in features_collect.keys():
                feats=torch.stack(features_collect[x])
                feat_mean=feats.mean(dim=0)
                features_mean.append(feat_mean)

                vars_2=(torch.stack(var_collect[x])**2).mean(dim=0)+(feats**2).mean(dim=0)-feat_mean**2
                vars_mean.append(torch.sqrt(vars_2))
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                vars_mean.append(torch.zeros_like(var_all[0]))
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named,torch.stack(vars_mean),var_all
    else:
        return features_all, labels_all, fnames_all, camids_all