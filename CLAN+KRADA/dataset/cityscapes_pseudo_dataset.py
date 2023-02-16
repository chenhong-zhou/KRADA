import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import random


class cityscapesSTDataSet(data.Dataset):
    def __init__(self, root, list_path, pseudo_root, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.pseudo_root = pseudo_root
        
        #self.mean_bgr = np.array([72.30608881, 82.09696889, 71.60167789])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            pseudo_label_file = osp.join(self.pseudo_root, "%s/%s" % (self.set, name))
            
          #  print('img_file: ', img_file)
            self.files.append({
                "img": img_file,
                "pseudo_label": pseudo_label_file,
                "name": name
            })
    
    def __scale__(self):
        cropsize = self.crop_size
        if self.scale:
            r = random.random()
            if r > 0.7:
                cropsize = (int(self.crop_size[0] * 1.1), int(self.crop_size[1] * 1.1))
            elif r < 0.3:
                cropsize = (int(self.crop_size[0] * 0.8), int(self.crop_size[1] * 0.8))

        return cropsize
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        cropsize = self.__scale__()
        
        try:
          #  print('datafiles["img"]', datafiles["img"])
          #  print('datafiles["pseudo_label"]', datafiles["pseudo_label"])
            image = Image.open(datafiles["img"]).convert('RGB')
            name = datafiles["name"]
            pseudo_label = Image.open(datafiles["pseudo_label"])
       #     print('image before', np.asarray(image, np.float32).shape)
            # resize
            image = image.resize(cropsize, Image.BICUBIC)
            image = np.asarray(image, np.float32)
        #    print('image', image.shape)
            size = image.shape
            image = image[:, :, ::-1]  # change to BGR
            image -= self.mean
            image = image.transpose((2, 0, 1))
         #   print('pseudo_label before', np.asarray(pseudo_label, np.float32).shape)
            pseudo_label = pseudo_label.resize(self.crop_size, Image.NEAREST)
            pseudo_label = np.asarray(pseudo_label, np.float32)
        #    print('pseudo_label', pseudo_label.shape)
            size_pl = pseudo_label.shape
    
            if self.is_mirror and random.random() < 0.5:
                idx = [i for i in range(size[1] - 1, -1, -1)]
                image = np.take(image, idx, axis = 2)
                idx_pl = [i for i in range(size_pl[1] - 1, -1, -1)]
                pseudo_label = np.take(pseudo_label, idx_pl, axis=1)



        except Exception as e:
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index)
        
        return image.copy(), pseudo_label.copy(), np.array(size), np.array(size), name


if __name__ == '__main__':
    dst = cityscapesSTDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=200)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
