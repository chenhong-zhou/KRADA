import os
from .cityscapes import cityscapesDataSet
from .cityscapes_pesudo_label import cityscapesSTDataSet
from .cityscapes_self_distill import cityscapesSelfDistillDataSet
from .synthia import synthiaDataSet
from .gta5 import GTA5DataSet

class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "gta5_train": {
            "data_dir": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/GTA5_synthetic/",
            "data_list": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/GTA5_synthetic/GTA5_train_unknown_2lei_58_list.txt"
        },
        "synthia_train": {
            "data_dir": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/",
            "data_list": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/RGB_unknown_3lei_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/",
            "data_list": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/split_train.txt"
        },
        "cityscapes_pesudo_label_train": {
            "data_dir": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/",
            "data_list": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/split_train.txt",
            "label_dir": "pesudo_label/train"
        },        
        "cityscapes_self_distill_train": {
            "data_dir": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/",
            "data_list": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/split_train.txt",
            "label_dir": "cityscapes/soft_labels/inference/cityscapes_train"
        },
        "cityscapes_val": {
            "data_dir": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/",
            "data_list": "/home/comp/cschzhou/Data/OSDA/DATA_Segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None):
        if "gta5" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTA5DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'pesudo' in name:
                args['label_dir'] = attrs["label_dir"]
                return cityscapesSTDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        raise RuntimeError("Dataset not available: {}".format(name))