import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
import os.path as osp
from PIL import Image

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier, build_classifier_1
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

import scipy.io as io    

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)

    if pixel_weights is None:   
        return torch.mean(torch.sum(loss, dim=1))    

    num = len(np.where(pixel_weights.cpu().numpy()==1)[0])      
    return torch.sum(pixel_weights*torch.sum(loss, dim=1))/num

def train(cfg, local_rank, distributed):
    logger = logging.getLogger("FADA.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)

    classifier_1 = build_classifier_1(cfg)
    classifier_1.to(device)
    
    model_D = build_adversarial_discriminator(cfg)
    model_D.to(device)

    if local_rank==0:
        print(feature_extractor)
        print(model_D)

    batch_size = cfg.SOLVER.BATCH_SIZE//2
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())//2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier_1 = torch.nn.parallel.DistributedDataParallel(
            classifier_1, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg3
        )        
        pg4 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg4
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    optimizer_cls_1 = torch.optim.SGD(classifier_1.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls_1.zero_grad()
    
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    start_epoch = 0
    iteration = 0
    
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        
        if "model_D" in checkpoint:
            logger.info("Loading model_D from {}".format(cfg.resume))
            model_D_weights = checkpoint['model_D'] if distributed else strip_prefix_if_present(checkpoint['model_D'], 'module.')
            model_D.load_state_dict(model_D_weights)
        if "classifier_1" not in checkpoint:
            logger.info("Loading classifier_1 from {}".format(cfg.resume))
            classifier_1_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
            classifier_1.load_state_dict(classifier_1_weights)    
        else:
            logger.info("Loading classifier_1 from {}".format(cfg.resume))
            classifier_1_weights = checkpoint['classifier_1'] if distributed else strip_prefix_if_present(checkpoint['classifier_1'], 'module.')
            classifier_1.load_state_dict(classifier_1_weights)               
    
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)

    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None
    
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size, shuffle=(src_train_sampler is None), num_workers=4, pin_memory=True, sampler=src_train_sampler, drop_last=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    weight_class = torch.ones(cfg.MODEL.NUM_CLASSES).cuda(non_blocking=True) 
    weight_class[cfg.MODEL.NUM_CLASSES - 1] = 0.2

 
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCELoss(reduction='none')
    criterion_weight = torch.nn.CrossEntropyLoss(ignore_index  = 255, weight = weight_class)

    max_iters = cfg.SOLVER.MAX_ITER
    source_label = 0
    target_label = 1
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    classifier_1.train()    
    model_D.train()
    start_training_time = time.time()
    end = time.time()

    class_prior = io.loadmat('./datasets/Synthia_source_prior_class_ratio.mat')  
    class_ratio = torch.from_numpy(class_prior['num_each_class_ratio']).cuda(non_blocking=True)

    for i, ((src_input, src_label, src_name), (tgt_input, tgt_PLable, tgt_name)) in enumerate(zip(src_train_loader, tgt_train_loader)):
        
        data_time = time.time() - end

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
        current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr*10
        for index in range(len(optimizer_cls_1.param_groups)):
            optimizer_cls_1.param_groups[index]['lr'] = current_lr*10            
        for index in range(len(optimizer_D.param_groups)):
            optimizer_D.param_groups[index]['lr'] = current_lr_D
            

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_cls_1.zero_grad()        
        optimizer_D.zero_grad()
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_PLable = tgt_PLable.cuda(non_blocking=True)


        
        src_size = src_input.shape[-2:]
        tgt_size = tgt_input.shape[-2:]

        src_fea = feature_extractor(src_input) 

        src_pred = classifier(src_fea, src_size)
        src_pred_1 = classifier_1(src_fea, src_size)


        temperature = 1.8
        src_pred = src_pred.div(temperature)
        src_pred_1 = src_pred_1.div(temperature)     



        interp_source = nn.Upsample(size=(src_size[0], src_size[1]), mode='bilinear', align_corners=True)
        interp_target = nn.Upsample(size=(tgt_size[0], tgt_size[1]), mode='bilinear', align_corners=True)


        if (len(torch.unique(tgt_PLable)) != 1):
            print('len(unk)', len(np.where(tgt_PLable.cpu().numpy()==(cfg.MODEL.NUM_CLASSES - 1))[0])) 
            up_img = nn.Upsample(size=(src_size[0], src_size[1]), mode='bilinear', align_corners=True)
            up_pl = nn.Upsample(size=(src_size[0], src_size[1]), mode='nearest')
            t_fea= feature_extractor(up_img(tgt_input))
            t_pred = classifier(t_fea, src_size)
            
            t_pred = t_pred.div(temperature)
            
            
            labels_all = torch.cat([src_label, up_pl(tgt_PLable.unsqueeze(1)).squeeze(1)], 0)
            pred_all = torch.cat([src_pred, t_pred], 0)
            labels_all = labels_all.cuda(non_blocking=True).long()
            
            loss1 = criterion_weight(pred_all, labels_all)
            loss2 = criterion(src_pred_1, src_label)
            loss_seg = (criterion_weight(pred_all, labels_all) + criterion(src_pred_1, src_label))/2.0        
        else:
            loss_seg = (criterion(src_pred, src_label) + criterion(src_pred_1, src_label) )/2.0

        loss_seg.backward()

     
        
        
        # generate soft labels
        src_soft_label_1 = F.softmax(src_pred_1, dim=1).detach()
        src_soft_label_1[src_soft_label_1>0.9] = 0.9
        
        tgt_fea = feature_extractor(tgt_input)
        tgt_pred_no_div = classifier(tgt_fea, tgt_size)
        tgt_pred_no_div_1 = classifier_1(tgt_fea, tgt_size)     


        
        tgt_pred = tgt_pred_no_div.div(temperature)
        tgt_pred_1 = tgt_pred_no_div_1.div(temperature)   

        tgt_soft_pred = F.softmax(tgt_pred, dim=1)  # tgt: C1  prediction
        tgt_soft_pred_1 = F.softmax(tgt_pred_1, dim=1) # tgt: C2 prediction


        tgt_soft_label = F.softmax(tgt_pred, dim=1)
        tgt_soft_label_1 = F.softmax(tgt_pred_1, dim=1)
        
        
        max_prob_target1_score_known = torch.max(tgt_soft_pred[:, :cfg.MODEL.NUM_CLASSES-1, :, :], 1)[0].unsqueeze(1) 
        max_prob_target1_score_unknown = tgt_soft_pred[:, cfg.MODEL.NUM_CLASSES-1, :, :].unsqueeze(1)     
        
        max_prob_target_score = torch.max(tgt_soft_pred_1, 1)[0].unsqueeze(1)   

        L2_norm_threshold = 0.1
        beta  = 0.99
        


        tmp1 = torch.softmax(tgt_pred_1, dim=1)  
        tmp2 = tmp1.permute([2,3,0,1]) 

        tmp3 = (tmp2 - class_ratio).permute([2,3,0,1])  
        L2_norm_map = torch.sum(tmp3.mul(tmp3), dim = 1) .unsqueeze(1)  
        
        
        zero = torch.zeros_like(max_prob_target_score) #known
        one = torch.ones_like(max_prob_target_score) #unk
        bg_all = torch.ones_like(max_prob_target_score)*255 
        unk_all = torch.ones_like(max_prob_target_score)*(cfg.MODEL.NUM_CLASSES-1)  
        
        
     ##################### source  L2_norm_map_source: start
        tmp1 = torch.softmax(src_pred_1.detach().cpu(), dim=1)
        tmp2 = tmp1.permute([2,3,0,1])
        tmp3 = (tmp2 - class_ratio.detach().cpu()).permute([2,3,0,1])
        L2_norm_map_source = torch.sum(tmp3.mul(tmp3), dim = 1).unsqueeze(1)
        N_source = torch.flatten(L2_norm_map_source).shape[0]
        sorted_norm_map_source, indices = torch.sort(torch.flatten(L2_norm_map_source))

        
        thred_00 = sorted_norm_map_source[int(N_source*0.0005)]
        thred_0 = sorted_norm_map_source[int(N_source*0.001)]
        thred_1 = sorted_norm_map_source[int(N_source*0.005)]
        thred_2 = sorted_norm_map_source[int(N_source*0.01)]
        thred_3 = sorted_norm_map_source[int(N_source*0.05)]           
        thred_4 = sorted_norm_map_source[int(N_source*0.1)]
        
        
################source pseudo_label_source :
        bg_all_source = torch.ones_like(L2_norm_map_source)*255 
        unk_all_source = torch.ones_like(L2_norm_map_source)*(cfg.MODEL.NUM_CLASSES-1)  
        

        pseudo_label_source = torch.where(L2_norm_map_source < L2_norm_threshold, unk_all_source, bg_all_source)           
        unk_num_source = len(np.where(pseudo_label_source.cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 

        pseudo_label_source_00 = torch.where(L2_norm_map_source < thred_00, unk_all_source, bg_all_source)           
        unk_num_source_00 = len(np.where(pseudo_label_source_00.cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 
        
        pseudo_label_source_0 = torch.where(L2_norm_map_source < thred_0, unk_all_source, bg_all_source)           
        unk_num_source_0 = len(np.where(pseudo_label_source_0.cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 

        pseudo_label_source_1 = torch.where(L2_norm_map_source < thred_1, unk_all_source, bg_all_source)           
        unk_num_source_1 = len(np.where(pseudo_label_source_1.cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 

        pseudo_label_source_2 = torch.where(L2_norm_map_source < thred_2, unk_all_source, bg_all_source)           
        unk_num_source_2 = len(np.where(pseudo_label_source_2.cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 

        pseudo_label_source_3 = torch.where(L2_norm_map_source < thred_3, unk_all_source, bg_all_source)           
        unk_num_source_3 = len(np.where(pseudo_label_source_3.cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 

        pseudo_label_source_4 = torch.where(L2_norm_map_source < thred_4, unk_all_source, bg_all_source)           
        unk_num_source_4 = len(np.where(pseudo_label_source_4.cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 
        
        
        # print('(thred_00) old & new L2_norm_threshold: ', L2_norm_threshold, L2_norm_threshold*beta + (1- beta)*thred_00)
        
        
        f_threshold = open(osp.join(output_dir,'threshold.txt'), 'a')
        f_threshold.write('{0:3d} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6:3f}   {7:3d} {8:3d} {9:3d} {10:3d} {11:3d} {12:3d}    {13:.3f} {14:.3f}   {15:.3f} {16:.3f} {17:.3f} {18:.3f} \n'.format(N_source, thred_00, thred_0, thred_1, thred_2, thred_3, thred_4, unk_num_source_00, unk_num_source_0, unk_num_source_1, unk_num_source_2, unk_num_source_3, unk_num_source_4, L2_norm_threshold, L2_norm_threshold*beta + (1- beta)*thred_00, torch.mean(L2_norm_map_source), torch.median(L2_norm_map_source), torch.max(L2_norm_map_source), torch.min(L2_norm_map_source)))
        f_threshold.close()
        


        del tmp1, tmp2, tmp3, L2_norm_map_source, N_source, sorted_norm_map_source, indices
        del bg_all_source, unk_all_source
        del pseudo_label_source, pseudo_label_source_0, pseudo_label_source_1, pseudo_label_source_2, pseudo_label_source_3, pseudo_label_source_4
        
        if (iteration < 2000):   
            L2_norm_threshold = 0.1
        else:
            L2_norm_threshold = L2_norm_threshold*beta + (1- beta)*thred_00
        #############################
        
        if (iteration < 2000):  
            Flag = 0
            pseudo_label_new = bg_all
            known_region_map = one
            pseudo_label_new2 = torch.where(max_prob_target1_score_unknown > max_prob_target1_score_known, unk_all, bg_all)   

            for  ii in range(pseudo_label_new.shape[0]):
            
                unk_num_1 = len(np.where(pseudo_label_new[ii,:,:,:].cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0])
                
                        
                unk_num_2 = len(np.where(pseudo_label_new2[ii,:,:,:].cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 
                
                num_unk = len(np.where(pseudo_label_new[ii,:,:,:].cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 
            #    print('pseudo_label_len', num_unk)

                f_map = open(osp.join(output_dir,'weight_map.txt'), 'a')
                f_map.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:4d} {6:4d} {7:4d} {8:4d}\n'.format(L2_norm_threshold, torch.mean(L2_norm_map), torch.median(L2_norm_map), torch.max(L2_norm_map), torch.min(L2_norm_map), num_unk, Flag, unk_num_1, unk_num_2))
                f_map.close()             

        else:  
            Flag = 1         
            pseudo_label_new = torch.where(L2_norm_map < L2_norm_threshold, unk_all, bg_all)   
            
            
            known_region_map = torch.where(pseudo_label_new == (cfg.MODEL.NUM_CLASSES-1), zero, one)
            
            pseudo_label_out =  torch.squeeze(pseudo_label_new).cpu().numpy()
            pseudo_label_out = np.asarray(pseudo_label_out, dtype=np.uint8) 


            pseudo_label_new2 = torch.where(max_prob_target1_score_unknown > max_prob_target1_score_known, unk_all, bg_all)   


            for  ii in range(pseudo_label_out.shape[0]):
                pseudo_label_out_single = Image.fromarray(pseudo_label_out[ii, :,:])
                save_pseudo_label_dir = tgt_name[ii].split('/')[0]
                
                if not os.path.exists(osp.join('./pesudo_label/train', "%s" % (save_pseudo_label_dir))):
                    os.makedirs(osp.join('./pesudo_label/train', "%s" % (save_pseudo_label_dir)))

                pseudo_label_out_single.save(osp.join('./pesudo_label/train', "%s/%s" % (save_pseudo_label_dir, tgt_name[ii].split('/')[1])))
            
                unk_num_1 = len(np.where(pseudo_label_new[ii,:,:,:].cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0])

                        
                unk_num_2 = len(np.where(pseudo_label_new2[ii,:,:,:].cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 

                num_unk = len(np.where(pseudo_label_new[ii,:,:,:].cpu().numpy()==(cfg.MODEL.NUM_CLASSES-1))[0]) 


                f_map = open(osp.join(output_dir,'weight_map.txt'), 'a')
                f_map.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:4d} {6:4d} {7:4d} {8:4d}\n'.format(L2_norm_threshold, torch.mean(L2_norm_map), torch.median(L2_norm_map), torch.max(L2_norm_map), torch.min(L2_norm_map), num_unk, Flag, unk_num_1, unk_num_2))
                f_map.close()             
        
        
        
        tgt_soft_label_1 = tgt_soft_label_1.detach()
        tgt_soft_label_1[tgt_soft_label_1>0.9] = 0.9
        
        tgt_D_pred = model_D(tgt_fea, tgt_size)


        loss_adv_tgt = 0.001*soft_label_cross_entropy(tgt_D_pred, torch.cat((tgt_soft_label_1, torch.zeros_like(tgt_soft_label_1)), dim=1), pixel_weights=torch.squeeze(known_region_map))
        loss_adv_tgt.backward()

        optimizer_fea.step()
        optimizer_cls.step()
        optimizer_cls_1.step()
        
        optimizer_D.zero_grad()
        
        src_D_pred = model_D(src_fea.detach(), src_size)
        loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label_1, torch.zeros_like(src_soft_label_1)), dim=1))
        loss_D_src.backward()

        tgt_D_pred = model_D(tgt_fea.detach(), tgt_size)
        loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label_1), tgt_soft_label_1), dim=1), pixel_weights=torch.squeeze(known_region_map.detach()))
        loss_D_tgt.backward()


        optimizer_D.step()
            
        meters.update(loss_seg=loss_seg.item())
        meters.update(loss_adv_tgt=loss_adv_tgt.item())
        meters.update(loss_D=(loss_D_src.item()+loss_D_tgt.item()))
        meters.update(loss_D_src=loss_D_src.item())
        meters.update(loss_D_tgt=loss_D_tgt.item())

        iteration = iteration + 1
        
        n = src_input.size(0)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer_fea.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                
        if (iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD==0) and save_to_disk:
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict(), 'classifier_1':classifier_1.state_dict(), 'model_D': model_D.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_cls': optimizer_cls.state_dict(), 'optimizer_cls_1': optimizer_cls_1.state_dict(),'optimizer_D': optimizer_D.state_dict()}, filename)

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
        

  
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor, classifier, classifier_1          

def run_test(cfg, feature_extractor, classifier, classifier_1, local_rank, distributed):
    logger = logging.getLogger("FADA.tester")
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    if distributed:
        feature_extractor, classifier, classifier_1 = feature_extractor.module, classifier.module, classifier_1.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    classifier_1.eval()    
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            output = classifier(feature_extractor(x), size)
            output = classifier_1(feature_extractor(x), size)
            output = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("FADA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    fea, cls = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, fea, cls, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
