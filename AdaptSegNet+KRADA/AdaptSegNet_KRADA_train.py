import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
from utils.loss import WeightedBCEWithLogitsLoss
from PIL import Image

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_pseudo_dataset import cityscapesSTDataSet
import scipy.io as io


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/'
DATA_LIST_PATH = './dataset/synthia_list/RGB_unknown_3cls_list.txt'


IGNORE_LABEL = 255
INPUT_SIZE = '1280,760'
DATA_DIRECTORY_TARGET = '/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/split_train.txt'
PSEUDO_LABEL_DIRECTORY_TARGET = './syn2city_split_pseudo_label/syn_pseudo_label'




INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 14
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9

RESTORE_FROM = './model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
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
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
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
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
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
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()




def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    weight_class = torch.ones(args.num_classes).cuda(args.gpu)  # 14
    weight_class[args.num_classes - 1] = 0.1  
    
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not args.num_classes == 14 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes-1)
    model_D2 = FCDiscriminator(num_classes=args.num_classes-1)

    model_D1.train()
    model_D1.cuda(args.gpu)

    model_D2.train()
    model_D2.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesSTDataSet(args.data_dir_target, args.data_list_target, args.pseudo_label_dir_target, max_iters=args.num_steps * args.iter_size * args.batch_size, crop_size=input_size_target,scale=False, mirror=args.random_mirror, mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    targetloader_iter = enumerate(targetloader)


    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1


    criterion = torch.nn.CrossEntropyLoss(ignore_index  = 255)
    criterion_weight = torch.nn.CrossEntropyLoss(ignore_index  = 255, weight = weight_class)

            
    class_prior = io.loadmat('/home/comp/cschzhou/Data/OSDA/CLAN-master_gai_new_focal_xin_2/syn2city_split_pseudo_label/Synthia_source_prior_class_ratio.mat')  
    class_ratio = torch.from_numpy(class_prior['num_each_class_ratio']).cuda(args.gpu)
    print('class_ratio', class_ratio)
    print('class_ratio.shape', class_ratio.shape)  #(1,13)
    
    L2_norm_threshold = 0.1
    beta  = 0.99

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = next(trainloader_iter)
            images_s, labels_s, _, _ = batch
            images_s = Variable(images_s).cuda(args.gpu)
            labels_s = Variable(labels_s).cuda(args.gpu)
            labels_s = Variable(labels_s.long())

            pred1, pred2, pred3, pred4 = model(images_s)        
            pred1 = interp(pred1)
            pred2 = interp(pred2)
            pred3 = interp(pred3)
            pred4 = interp(pred4)

            
            
            # train with target
            _, batch = next(targetloader_iter)
            images_t, pseudo_label_t, _,  name = batch
            images_t = Variable(images_t).cuda(args.gpu)
            pseudo_label_t = Variable(pseudo_label_t).cuda(args.gpu)

            if (len(torch.unique(pseudo_label_t)) != 1):
                print('len(unk)', len(np.where(pseudo_label_t.cpu().numpy()==(args.num_classes-1))[0])) #unk:1 
                up_img = nn.Upsample(size=(images_s.shape[-2], images_s.shape[-1]), mode='bilinear', align_corners=True)
                up_pl = nn.Upsample(size=(labels_s.shape[-2], labels_s.shape[-1]), mode='nearest')
                pred_t1, pred_t2, _, _ = model(up_img(images_t))
                pred_t1 = interp(pred_t1) # 1 14
                pred_t2 = interp(pred_t2)
                
                labels_all = torch.cat([labels_s, up_pl(pseudo_label_t.unsqueeze(1)).squeeze(1)], 0)
                pred_all_1 = torch.cat([pred1, pred_t1], 0)
                pred_all_2 = torch.cat([pred2, pred_t2], 0)                
                labels_all = Variable(labels_all.long())
            
                
                loss_seg1 = (criterion_weight(pred_all_1, labels_all) + criterion(pred3, labels_s))/2
                
                loss_seg2 = (criterion_weight(pred_all_2, labels_all) + criterion(pred4, labels_s))/2
                loss = loss_seg2 + args.lambda_seg * loss_seg1                
            else:
                loss_seg1 = (criterion(pred1, labels_s) + criterion(pred3, labels_s))/2
                loss_seg2 = (criterion(pred2, labels_s) + criterion(pred4, labels_s))/2
                loss = loss_seg2 + args.lambda_seg * loss_seg1
            

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.item() / args.iter_size
            loss_seg_value2 += loss_seg2.item() / args.iter_size


            pred_target1, pred_target2, pred_target3, pred_target4 = model(images_t)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)
            pred_target3 = interp_target(pred_target3)
            pred_target4 = interp_target(pred_target4)
            
            D_out3 = interp_target(model_D1(F.softmax(pred_target3, dim=1)))
            D_out4 = interp_target(model_D2(F.softmax(pred_target4, dim=1)))
            

######################   C1 pred_target2        
            soft_prob_t2 = F.softmax(pred_target2, dim = 1)      
            
            max_prob_target1_score_known = torch.max(soft_prob_t2[:, :args.num_classes-1, :, :], 1)[0].unsqueeze(1) 
            max_prob_target1_score_unknown = soft_prob_t2[:, args.num_classes-1, :, :].unsqueeze(1)
            
            #pred4 C2: 
            max_prob_target_score = torch.max(F.softmax(pred_target4, dim = 1), 1)[0].unsqueeze(1)   
            
            

            tmp1 = torch.softmax(pred_target4, dim=1) 
            tmp2 = tmp1.permute([2,3,0,1]) 
            tmp3 = (tmp2 - class_ratio).permute([2,3,0,1]) 
            L2_norm_map = torch.sum(tmp3.mul(tmp3), dim = 1) .unsqueeze(1)
            print('L2_norm_map', L2_norm_map.shape)
 
        
            zero = torch.zeros_like(L2_norm_map) #known
            one = torch.ones_like(L2_norm_map) #unk
            bg_all = torch.ones_like(L2_norm_map)*255 
            unk_all = torch.ones_like(L2_norm_map)*(args.num_classes-1) 


         ##################### source  L2_norm_map_source: start
            tmp1 = torch.softmax(pred4.detach().cpu(), dim=1)
            tmp2 = tmp1.permute([2,3,0,1])
            tmp3 = (tmp2 - class_ratio.detach().cpu()).permute([2,3,0,1])#class_ratio: (1,13)
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
            
            print('N_source, thred_0, thred_1, thred_2, thred_3, thred_4: ', N_source, thred_0, thred_1, thred_2, thred_3, thred_4)
            
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
            # print('unk_num_source, _00,  _0, _1, _2, _3, _4: ', unk_num_source, unk_num_source_00, unk_num_source_0, unk_num_source_1, unk_num_source_2, unk_num_source_3, unk_num_source_4)
            
            print('(thred_00) old & new L2_norm_threshold: ', L2_norm_threshold, L2_norm_threshold*beta + (1- beta)*thred_00)
        
            
            f_threshold = open(osp.join(args.snapshot_dir,'threshold.txt'), 'a')
            f_threshold.write('{0:3d} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6:3f}   {7:3d} {8:3d} {9:3d} {10:3d} {11:3d} {12:3d}    {13:.3f} {14:.3f}   {15:.3f} {16:.3f} {17:.3f} {18:.3f} \n'.format(N_source, thred_00, thred_0, thred_1, thred_2, thred_3, thred_4, unk_num_source_00, unk_num_source_0, unk_num_source_1, unk_num_source_2, unk_num_source_3, unk_num_source_4, L2_norm_threshold, L2_norm_threshold*beta + (1- beta)*thred_00, torch.mean(L2_norm_map_source), torch.median(L2_norm_map_source), torch.max(L2_norm_map_source), torch.min(L2_norm_map_source)))
            f_threshold.close()
            


            del tmp1, tmp2, tmp3, L2_norm_map_source, N_source, sorted_norm_map_source, indices
            del bg_all_source, unk_all_source
            del pseudo_label_source, pseudo_label_source_00, pseudo_label_source_0, pseudo_label_source_1, pseudo_label_source_2, pseudo_label_source_3, pseudo_label_source_4

            
            #####################
            if (i_iter < 2000):  
                L2_norm_threshold = 0.1
            else:
                L2_norm_threshold = L2_norm_threshold*beta + (1- beta)*thred_00
            ############################################### source:   end    update L2_norm_threshold:  end
            
            
            pseudo_label_new = torch.where(L2_norm_map < L2_norm_threshold, unk_all, bg_all)           
            unk_num_1 = len(np.where(pseudo_label_new.cpu().numpy()==(args.num_classes-1))[0])         
            known_region_map = torch.where(pseudo_label_new == (args.num_classes-1), zero, one)
            print('known_region_map.shape', known_region_map.shape)
            
            
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
############                
            
            
            
            loss_adv_target1 = weighted_bce_loss(D_out3,
                                       Variable(torch.FloatTensor(D_out3.data.size()).fill_(source_label)).cuda(
                                           args.gpu), known_region_map)

            loss_adv_target2 = weighted_bce_loss(D_out4,
                                        Variable(torch.FloatTensor(D_out4.data.size()).fill_(source_label)).cuda(
                                            args.gpu), known_region_map)

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size

            # train D
            
            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True


            pred3 = pred3.detach()
            pred4 = pred4.detach()
            
            D_out3 = model_D1(F.softmax(pred3, dim=1))
            D_out4 = model_D2(F.softmax(pred4, dim=1))
            
            loss_D1 = bce_loss(D_out3,
                              Variable(torch.FloatTensor(D_out3.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out4,
                               Variable(torch.FloatTensor(D_out4.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target3 = pred_target3.detach()
            pred_target4 = pred_target4.detach()
            
            D_out3 = interp_target(model_D1(F.softmax(pred_target3, dim=1)))
            D_out4 = interp_target(model_D2(F.softmax(pred_target4, dim=1)))
 

            loss_D1 = weighted_bce_loss(D_out3,
                              Variable(torch.FloatTensor(D_out3.data.size()).fill_(target_label)).cuda(args.gpu), known_region_map)

            loss_D2 = weighted_bce_loss(D_out4,
                               Variable(torch.FloatTensor(D_out4.data.size()).fill_(target_label)).cuda(args.gpu), known_region_map)

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))

        f_loss = open(osp.join(args.snapshot_dir,'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}\n'.format(
            loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))
        f_loss.close()

        f_map = open(osp.join(args.snapshot_dir,'weight_map.txt'), 'a')
        f_map.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:4d} {6:4d} {7:4d} {8:4d}\n'.format( L2_norm_threshold, torch.mean(L2_norm_map), torch.median(L2_norm_map), torch.max(L2_norm_map), torch.min(L2_norm_map), num_unk, Flag, unk_num_1, unk_num_2))
        f_map.close()
        
        
        if i_iter >= args.num_steps_stop - 1:
            print ('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'syn_' + str(i_iter) + '_D2.pth'))


if __name__ == '__main__':
    main()
