import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class cityscapesSTDataSet(data.Dataset):
    def __init__(
        self,
        data_root,
        data_list,
        label_dir,
        max_iters=None,
        num_classes=14, 
        split="train",
        transform=None,
        ignore_label=255,
        debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.label_dir = label_dir
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()

        for fname in content:
            name = fname.strip()
        #    print(name)
        #    
            if self.split in  ["train"]:
                label_dir_all = os.path.join(self.label_dir, name)    
            #    print('label_dir_all', label_dir_all)        
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.data_root, "leftImg8bit/%s/%s" % (self.split, name)
                    ),
                    "label": label_dir_all,
                    "name": name,
                }
            )
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            13: 13,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "fence",
            4: "pole",
            5: "sign",
            6: "vegetation",
            7: "sky",
            8: "person",
            9: "rider",
            10: "car",
            11: "motocycle",
            12: "bicycle",
            13: "unknown",
        }
        
    
            
        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]),dtype=np.float32)
        name = datafiles["name"]

        # # re-assign labels to match the format of Cityscapes
        # label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        # for k in self.trainid2name.keys():
            # label_copy[label == k] = k
        # label = Image.fromarray(label_copy)
        
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, name
