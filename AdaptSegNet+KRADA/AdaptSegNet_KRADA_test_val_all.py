import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model.deeplab_multi import DeeplabMulti
from dataset.cityscapes_dataset import cityscapesDataSet
from torch.utils import data
import os
from PIL import Image
import torch.nn as nn
import os.path as osp                     
import torch.nn.functional as F

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/'
DATA_LIST_PATH = '/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val.txt'
SAVE_PATH = './snapshots/models'
IGNORE_LABEL = 255
NUM_CLASSES = 14
#NUM_STEPS = 500 # Number of images in the validation set.
###########################
SET = 'val'


def create_map(input_size, mode):
    if mode == 'h':
        T_base = torch.arange(0, float(input_size[1]))
        T_base = T_base.view(input_size[1], 1)
        T = T_base
        for i in range(input_size[0] - 1):
            T = torch.cat((T, T_base), 1)
        T = torch.div(T, float(input_size[1]))
    if mode == 'w':
        T_base = torch.arange(0, float(input_size[0]))
        T_base = T_base.view(1, input_size[0])
        T = T_base
        for i in range(input_size[1] - 1):
            T = torch.cat((T, T_base), 0)
        T = torch.div(T, float(input_size[0]))
    T = T.view(1, 1, T.size(0), T.size(1))
    return T


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

##########iou bulk #####################################################
import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import csv

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def compute_mIoU(pred_imgs, gt_imgs, json_path):
    """
    Function to compute mean IoU
    Args:
    	pred_imgs: Predictions obtained using our Neural Networks
    	gt_imgs: Ground truth label maps
    	json_path: Path to cityscapes_info.json file
    Returns:
    	Mean IoU score
    """
    with open(json_path, 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])

    name_classes = np.array(info['label'], dtype=np.str)

    hist = np.zeros((num_classes, num_classes))

    for ind in range(len(gt_imgs)):
        pred = pred_imgs[ind]
        label = gt_imgs[ind]
        label_f = label.flatten()
        pred_f = pred.flatten()
        non_bg_ind = np.where(label_f != 255)
        label_f_ignored = label_f[non_bg_ind]
        pred_f_ignored = pred_f[non_bg_ind]


        if len(label_f_ignored) != len(pred_f_ignored):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(len(label_f_ignored), len(pred_f_ignored)))
            continue

        hist += fast_hist(label_f, pred_f, num_classes)

    mIoUs = per_class_iu(hist)

    return mIoUs


# def main():
#     """Create the model and start the evaluation process."""

for i in range(1, 51):
    
    args = get_arguments()
    
    model_path = args.save_path + '/syn_{0:d}.pth'.format(i*2000)
    print('model_path: ', model_path)

    filename_prob1 = args.save_path + '/p1.txt'

    
    f_mIoUs_prob1 = open(filename_prob1, 'a')
    f_mIoUs_prob1.write('model_path: '+model_path+'\n')
    f_mIoUs_prob1.write('Prob C1\n')
    f_mIoUs_prob1.close()     


    gpu0 = args.gpu



    model = DeeplabMulti(num_classes=args.num_classes)
    print('args.num_classes', args.num_classes)

    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024,512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    label_trues, label_preds_0, label_preds_1 = [], [], []
    label_preds_2, label_preds_3, label_preds_4, label_preds_5, label_preds_6 = [], [], [], [], []
    label_preds_7, label_preds_8, label_preds_9, label_preds_10 = [], [], [], []
    label_preds_zhenghe = []
    label_preds_prob1, label_preds_prob2 = [], []

  
    devkit_dir = './dataset/cityscapes_list'

    gt_dir = '/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/gtFine_trainvaltest/gtFine/val_unknown_3lei_0_12_unk_255'
    info_json_fn = join(devkit_dir, 'synthia2cityscapes_info.json')




    with torch.no_grad():
        for index, batch in enumerate(testloader):

            image, _, name = batch
            _, output2, _, output4  = model(Variable(image).cuda(gpu0))
            
            name = name[0].replace('leftImg8bit', 'gtFine_label16IDs')
            label_file = osp.join(gt_dir, "%s" % name)
            label = np.array(Image.open(label_file))
            label_trues.append(label) 

            output1 = interp(output2).data[0]
            output2 = interp(output4).data[0]   


            prob_ind1 = torch.max(output1, 0)[1] 
            prob_ind2 = torch.max(output2, 0)[1] 
            
            label_preds_prob1.append(prob_ind1.cpu().numpy().squeeze()) 
            
            out_prob1 = F.softmax(output1, dim = 0)
            max_output1_score_known = torch.max(out_prob1[:args.num_classes-1, :, :], 0)[0] 
            max_output1_score_unknown = out_prob1[args.num_classes-1, :, :] 
            
            unk_all = torch.ones_like(prob_ind1)*(args.num_classes-1)
            
            
            
    with open(info_json_fn, 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
                                                                                 
    name_classes = np.array(info['label'], dtype=np.str)



    mIoUs_prob1 = compute_mIoU(label_preds_prob1, label_trues, info_json_fn)    
    f_mIoUs_prob1 = open(filename_prob1, 'a')
    f_mIoUs_prob1.write('mIoU shape: {0:.2f} \n'.format(len(mIoUs_prob1)))
    f_mIoUs_prob1.write('Num classes include unknown: {0:.2f} \n'.format(num_classes))
    for ind_class in range(num_classes):
        f_mIoUs_prob1.write('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs_prob1[ind_class] * 100, 2)) +'\n')
    f_mIoUs_prob1.write('===> mIoU: ' + str(round(np.nanmean(mIoUs_prob1) * 100, 2))+'\n')
    f_mIoUs_prob1.write('===> mIoU exclude unknown class: ' + str(round(np.nanmean(mIoUs_prob1[:-1]) * 100, 2))+'\n')    
    f_mIoUs_prob1.close()


