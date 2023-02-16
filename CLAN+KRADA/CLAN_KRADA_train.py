import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
from PIL import Image
from model.CLAN_G import Res_Deeplab
from model.CLAN_D import FCDiscriminator

from utils.loss import CrossEntropyLoss2d
from utils.loss import WeightedBCEWithLogitsLoss

from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_pseudo_dataset import cityscapesSTDataSet

import scipy.io as io

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 2

IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 14  #syn2cityscape: 13 known classes + unknown class
RESTORE_FROM = './model/DeepLab_resnet_pretrained_init-f81d91e8.pth'

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP/20)
POWER = 0.9


SOURCE = 'SYNTHIA'
TARGET = 'cityscapes'
SET = 'train'



INPUT_SIZE_SOURCE = '1280,760'
DATA_DIRECTORY = '/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/'
DATA_LIST_PATH = './dataset/synthia_list/RGB_unknown_3cls_list.txt'
Lambda_weight = 0.01
Lambda_adv = 0.001
Lambda_local = 10
Epsilon = 0.4


INPUT_SIZE_TARGET = '1024,512'
DATA_DIRECTORY_TARGET = '/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/split_train.txt'
PSEUDO_LABEL_DIRECTORY_TARGET = './syn2city_split_pseudo_label/syn_pseudo_label'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : GTA5, SYNTHIA")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--pseudo-label-dir-target", type=str, default=PSEUDO_LABEL_DIRECTORY_TARGET,
                        help="Path to the pseudo file listing the images in the target dataset.")     
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, weight):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    criterion = CrossEntropyLoss2d(weight).cuda()    
    return criterion(pred, label)
    
    
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
    (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
    return output



def main():
    """Create the model and start the training."""
    weight_class = torch.ones(args.num_classes).cuda(args.gpu)  
    weight_class[args.num_classes - 1] = 0.03  
    
    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True
    
    class_prior = io.loadmat('./dataset/Synthia_source_prior_class_ratio.mat')  
    class_ratio = torch.from_numpy(class_prior['num_each_class_ratio']).cuda(args.gpu)
    print('class_ratio', class_ratio)
    print('class_ratio.shape', class_ratio.shape)  

    # Create Network
    model = Res_Deeplab(num_classes=args.num_classes)
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not args.num_classes == 14 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    if args.restore_from[:4] == './mo':        
        model.load_state_dict(new_params)
    else:
        model.load_state_dict(saved_state_dict)
        
    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # Init D
    model_D = FCDiscriminator(num_classes=args.num_classes-1)
    
    
    model_D.train()
    model_D.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    
    trainloader = data.DataLoader(
        SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size_source,
                    scale=True, mirror=True, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesSTDataSet(args.data_dir_target, args.data_list_target, args.pseudo_label_dir_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=True, mirror=True, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    
    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    criterion = torch.nn.CrossEntropyLoss(ignore_index  = 255)
    criterion_weight = torch.nn.CrossEntropyLoss(ignore_index  = 255, weight = weight_class)
    
    L2_norm_threshold = 0.1
    beta  = 0.99
    
    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)
        
        damping = (1 - i_iter/NUM_STEPS)

        #======================================================================================
        # train G
        #======================================================================================

        #Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s = Variable(images_s).cuda(args.gpu)
        labels_s = Variable(labels_s).cuda(args.gpu)

    

        # Train with Target
        _, batch = next(targetloader_iter)
        images_t, pseudo_label_t, _, _, name = batch
        images_t = Variable(images_t).cuda(args.gpu)
        pseudo_label_t = Variable(pseudo_label_t).cuda(args.gpu)

        pred_source1, pred_source2 = model(images_s)
        pred_source1 = interp_source(pred_source1) # 14
        pred_source2 = interp_source(pred_source2) # 13
        
        
        labels_s = Variable(labels_s.long())
        if (len(torch.unique(pseudo_label_t)) != 1): 
            print('len(unk)', len(np.where(pseudo_label_t.cpu().numpy()==(args.num_classes-1))[0])) #unk:1 
            up_img = nn.Upsample(size=(images_s.shape[-2], images_s.shape[-1]), mode='bilinear', align_corners=True)
            up_pl = nn.Upsample(size=(labels_s.shape[-2], labels_s.shape[-1]), mode='nearest')
            pred_t1, _ = model(up_img(images_t))
            pred_t1 = interp_source(pred_t1)
            
            labels_all = torch.cat([labels_s, up_pl(pseudo_label_t.unsqueeze(1)).squeeze(1)], 0)
            pred_all = torch.cat([pred_source1, pred_t1], 0)
            
            labels_all = Variable(labels_all.long())
            
            loss1 = criterion_weight(pred_all, labels_all)
            loss2 = criterion(pred_source2, labels_s)
            loss_seg = (criterion_weight(pred_all, labels_all) + criterion(pred_source2, labels_s))/2.0        
        else:
            loss1 = criterion(pred_source1, labels_s)
            loss2 = 0
            loss_seg = (criterion(pred_source1, labels_s) + criterion(pred_source2, labels_s))/2.0
        
        loss_seg.backward()

        

        pred_target1, pred_target2 = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2) 
        weight_map = weightmap(F.softmax(pred_target1[:, :args.num_classes-1, :, :], dim = 1), F.softmax(pred_target2, dim = 1))
        
        D_out = interp_target(model_D(F.softmax(pred_target1[:, :args.num_classes-1, :, :] + pred_target2, dim = 1)))
        
        soft_prob_t1 = F.softmax(pred_target1, dim = 1)      
        
        max_prob_target1_score_known = torch.max(soft_prob_t1[:, :args.num_classes-1, :, :], 1)[0].unsqueeze(1) 
        max_prob_target1_score_unknown = soft_prob_t1[:, args.num_classes-1, :, :].unsqueeze(1)
        
         
        max_prob_target_score = torch.max(F.softmax(pred_target2, dim = 1), 1)[0].unsqueeze(1)
        
        
      #  print('weight_map', weight_map.shape) # 1 1 512 1024
        
        tmp1 = torch.softmax(pred_target2, dim=1) 
        
        tmp2 = tmp1.permute([2,3,0,1]) 
        tmp3 = (tmp2 - class_ratio).permute([2,3,0,1])  
        L2_norm_map = torch.sum(tmp3.mul(tmp3), dim = 1).unsqueeze(1)
        print('L2_norm_map.shape', L2_norm_map.shape)  

        zero = torch.zeros_like(L2_norm_map) #known
        one = torch.ones_like(L2_norm_map)   #unk
        bg_all = torch.ones_like(L2_norm_map)*255 
        unk_all = torch.ones_like(L2_norm_map)*(args.num_classes-1)  
        

     ##################### source  L2_norm_map_source: start
        tmp1 = torch.softmax(pred_source2.detach().cpu(), dim=1)
        tmp2 = tmp1.permute([2,3,0,1])
        tmp3 = (tmp2 - class_ratio.detach().cpu()).permute([2,3,0,1]) 
        L2_norm_map_source = torch.sum(tmp3.mul(tmp3), dim = 1).unsqueeze(1)
        N_source = torch.flatten(L2_norm_map_source).shape[0]
        sorted_norm_map_source, indices = torch.sort(torch.flatten(L2_norm_map_source))
        # print('L2_norm_map_source.shape: ', L2_norm_map_source.shape)  
        # print('sorted_norm_map_source.shape: ', sorted_norm_map_source.shape)
        
        thred_00 = sorted_norm_map_source[int(N_source*0.0005)]
        thred_0 = sorted_norm_map_source[int(N_source*0.001)]
        thred_1 = sorted_norm_map_source[int(N_source*0.005)]
        thred_2 = sorted_norm_map_source[int(N_source*0.01)]
        thred_3 = sorted_norm_map_source[int(N_source*0.05)]           
        thred_4 = sorted_norm_map_source[int(N_source*0.1)]
        
        
        print('N_source, thred_00, thred_0, thred_1, thred_2, thred_3, thred_4: ', N_source, thred_00, thred_0, thred_1, thred_2, thred_3, thred_4)
        
################source pseudo_label_source :
        bg_all_source = torch.ones_like(L2_norm_map_source)*255 
        unk_all_source = torch.ones_like(L2_norm_map_source)*(args.num_classes-1)  

        pseudo_label_source = torch.where(L2_norm_map_source < L2_norm_threshold, unk_all_source, bg_all_source)           
        unk_num_source = len(np.where(pseudo_label_source.cpu().numpy()==(args.num_classes-1))[0]) 

        pseudo_label_source_00 = torch.where(L2_norm_map_source < thred_00, unk_all_source, bg_all_source)           
        unk_num_source_00 = len(np.where(pseudo_label_source_00.cpu().numpy()==(args.num_classes-1))[0]) 
        
        pseudo_label_source_0 = torch.where(L2_norm_map_source < thred_0, unk_all_source, bg_all_source)           
        unk_num_source_0 = len(np.where(pseudo_label_source_0.cpu().numpy()==(args.num_classes-1))[0]) 

        pseudo_label_source_1 = torch.where(L2_norm_map_source < thred_1, unk_all_source, bg_all_source)           
        unk_num_source_1 = len(np.where(pseudo_label_source_1.cpu().numpy()==(args.num_classes-1))[0]) 

        pseudo_label_source_2 = torch.where(L2_norm_map_source < thred_2, unk_all_source, bg_all_source)           
        unk_num_source_2 = len(np.where(pseudo_label_source_2.cpu().numpy()==(args.num_classes-1))[0]) 

        pseudo_label_source_3 = torch.where(L2_norm_map_source < thred_3, unk_all_source, bg_all_source)           
        unk_num_source_3 = len(np.where(pseudo_label_source_3.cpu().numpy()==(args.num_classes-1))[0]) 

        pseudo_label_source_4 = torch.where(L2_norm_map_source < thred_4, unk_all_source, bg_all_source)           
        unk_num_source_4 = len(np.where(pseudo_label_source_4.cpu().numpy()==(args.num_classes-1))[0]) 
        
        # print('pseudo_label_source_4: ', pseudo_label_source_4.shape)
        # print('unk_num_source, _00, _0, _1, _2, _3, _4: ', unk_num_source, unk_num_source_00, unk_num_source_0, unk_num_source_1, unk_num_source_2, unk_num_source_3, unk_num_source_4)
        
        # print('(thred_00) old & new L2_norm_threshold: ', L2_norm_threshold, L2_norm_threshold*beta + (1- beta)*thred_00)
        
        
        f_threshold = open(osp.join(args.snapshot_dir,'threshold.txt'), 'a')
        f_threshold.write('{0:3d} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6:3d} {7:3d} {8:3d} {9:3d} {10:3d}  {11:.3f} {12:.3f}   {13:.3f} {14:.3f} {15:.3f} {16:.3f} \n'.format(N_source, thred_0, thred_1, thred_2, thred_3, thred_4, unk_num_source_0, unk_num_source_1, unk_num_source_2, unk_num_source_3, unk_num_source_4, L2_norm_threshold, L2_norm_threshold*beta + (1- beta)*thred_00, torch.mean(L2_norm_map_source), torch.median(L2_norm_map_source), torch.max(L2_norm_map_source), torch.min(L2_norm_map_source)))
        f_threshold.close()
        


        del tmp1, tmp2, tmp3, L2_norm_map_source, N_source, sorted_norm_map_source, indices
        del bg_all_source, unk_all_source
        del pseudo_label_source, pseudo_label_source_0, pseudo_label_source_1, pseudo_label_source_2, pseudo_label_source_3, pseudo_label_source_4

        #####################
        if (i_iter < 2000):  
            L2_norm_threshold = 0.1
        else:
            L2_norm_threshold = L2_norm_threshold*beta + (1- beta)*thred_00
            

        ############################################### source:   end    update L2_norm_threshold:  end
     
        pseudo_label_new = torch.where(L2_norm_map < L2_norm_threshold, unk_all, bg_all)           
        unk_num_1 = len(np.where(pseudo_label_new.cpu().numpy()==(args.num_classes-1))[0])         
        known_region_map = torch.where(pseudo_label_new == (args.num_classes-1), zero, one)


  ####################
        pseudo_label_new2 = torch.where(max_prob_target1_score_unknown > max_prob_target1_score_known, unk_all, bg_all)           
        unk_num_2 = len(np.where(pseudo_label_new2.cpu().numpy()==(args.num_classes-1))[0])   
        
        
        
        
        if (i_iter < 2000):    
            Flag = 0
            pseudo_label_new = bg_all
            known_region_map = one
        else:  
            Flag = 1                       
            pseudo_label_out =  torch.squeeze(pseudo_label_new).cpu().numpy()
            pseudo_label_out = np.asarray(pseudo_label_out, dtype=np.uint8)      
            pseudo_label_out = Image.fromarray(pseudo_label_out)
            save_pseudo_label_dir = name[0].split('/')[0]

            if not os.path.exists(osp.join(args.pseudo_label_dir_target, "%s/%s" % (args.set, save_pseudo_label_dir))):
                os.makedirs(osp.join(args.pseudo_label_dir_target, "%s/%s" % (args.set, save_pseudo_label_dir)))

            pseudo_label_out.save(osp.join(args.pseudo_label_dir_target, "%s/%s/%s.png" % (args.set, save_pseudo_label_dir, name[0].split('/')[1].split('.')[0])))
            
        

        
        num_unk = len(np.where(pseudo_label_new.cpu().numpy()==(args.num_classes-1))[0]) 
        print('pseudo_label_len', num_unk)

        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out, 
                                    Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local, known_region_map)
        else:
            loss_adv = bce_loss(D_out,
                          Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_adv = loss_adv * Lambda_adv * damping
        loss_adv.backward()
        
        
        loss_weight = 0
        #======================================================================================
        # train D
        #======================================================================================
        
        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True
            
        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()
        
        D_out_s = interp_source(model_D(F.softmax(pred_source1[:, :args.num_classes-1, :, :] + pred_source2, dim = 1)))
        
        loss_D_s = bce_loss(D_out_s,
                          Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_D_s.backward()
        
        
        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        weight_map = weight_map.detach()
        known_region_map = known_region_map.detach()
        
        D_out_t = interp_target(model_D(F.softmax(pred_target1[:, :args.num_classes-1, :, :] + pred_target2, dim = 1)))
        
        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS): 
            loss_D_t = weighted_bce_loss(D_out_t, 
                                    Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local, known_region_map)
        else:
            loss_D_t = bce_loss(D_out_t,
                          Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(args.gpu))
            
        loss_D_t.backward()

        optimizer.step()
        optimizer_D.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'.format(
            i_iter, args.num_steps, loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))

        f_loss = open(osp.join(args.snapshot_dir,'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f}\n'.format(
            loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t, loss1, loss2))
        f_loss.close()

        f_map = open(osp.join(args.snapshot_dir,'weight_map.txt'), 'a')
        f_map.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:4d} {6:4d} {7:4d} {8:4d}\n'.format( L2_norm_threshold, torch.mean(L2_norm_map), torch.median(L2_norm_map), torch.max(L2_norm_map), torch.min(L2_norm_map), num_unk, Flag, unk_num_1, unk_num_2))
        f_map.close()


        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(args.num_steps) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(i_iter) + '_D.pth'))

if __name__ == '__main__':
    main()
