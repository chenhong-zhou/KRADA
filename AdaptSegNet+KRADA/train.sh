
CUDA_VISIBLE_DEVICES=3 python AdaptSegNet_KRADA_train.py --snapshot-dir ./snapshots/models --lambda-seg 0.1 --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001

