import argparse
import os
import math
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Generate label stat info")
parser.add_argument("-d",
        "--datadir",
        default="",
        help="path to load data",
        type=str,
    )
parser.add_argument("-n",
        "--nprocs",
        default=16,
        help="Number of processes",
        type=int,
    )
parser.add_argument("-o",
        "--output_dir",
        default="",
        help="path to save label info",
        type=str,
    )
args = parser.parse_args()
imgdir = os.path.join(args.datadir, 'images_unknown_2lei_58_0_16_255')
labdir = os.path.join(args.datadir, 'labels19_unknown_2lei_58_0_16_255')
labfiles = os.listdir(labdir)
nprocs = args.nprocs
savedir = args.output_dir

#17lei
ignore_label = 255
id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                              6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                              13: 13, 14: 14, 15: 15, 16: 16}

def generate_label_info():
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e:[] for e in os.listdir(imgdir)}

    for labfile in tqdm(labfiles):
        label = np.unique(np.array(Image.open(os.path.join(labdir, labfile)), dtype=np.float32))
        for lab in label:
            if lab in id_to_trainid.keys():
                l = id_to_trainid[lab]
                label_to_file[l].append(labfile)
                file_to_label[labfile].append(l)
    
    return label_to_file, file_to_label

def _foo(i):
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = dict()
    labfile = labfiles[i]
    file_to_label[labfile] = []
    label = np.unique(np.array(Image.open(os.path.join(labdir, labfile)), dtype=np.float32))
    for lab in label:
        if lab in id_to_trainid.keys():
            l = id_to_trainid[lab]
            label_to_file[l].append(labfile)
            file_to_label[labfile].append(l)
    return label_to_file, file_to_label


def main():
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e:[] for e in os.listdir(imgdir)}
    
    if nprocs==1:
        label_to_file, file_to_label = generate_label_info()
    else:
        with Pool(nprocs) as p:
            r = list(tqdm(p.imap(_foo, range(len(labfiles))), total=len(labfiles)))
        for l2f, f2l in r:
            for lab in range(len(l2f)):
                label_to_file[lab].extend(l2f[lab])
            for fname in f2l.keys():
                file_to_label[fname].extend(f2l[fname])
    with open(os.path.join(savedir, 'gtav_label_info.p'), 'wb') as f:
        pickle.dump((label_to_file, file_to_label), f)




if __name__ == "__main__":
    main()

