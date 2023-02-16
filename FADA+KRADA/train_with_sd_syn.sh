# train on source data
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_src.py -cfg configs/deeplabv2_r101_src_synthia.yaml OUTPUT_DIR results/src_r101_try_syn/

# train with fine-grained adversarial alignment
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 train_adv.py -cfg configs/deeplabv2_r101_adv_synthia_14.yaml OUTPUT_DIR results/adv_L2_norm resume results/src_r101_try_syn/model_iter020000.pth

# generate pseudo labels for self distillation
CUDA_VISIBLE_DEVICES=0 python test.py -cfg configs/deeplabv2_r101_adv_synthia_14.yaml --saveres resume results/adv_L2_norm/model_iter022000.pth OUTPUT_DIR datasets/cityscapes/soft_labels DATASETS.TEST cityscapes_train

# train with self distillation
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_self_distill.py -cfg configs/deeplabv2_r101_tgt_synthia_self_distill_14.yaml OUTPUT_DIR results/sd_test_syn





#########test:
#adv_model: 

CUDA_VISIBLE_DEVICES=0 python test_adv_unk_C1_log.py -cfg configs/deeplabv2_r101_adv_synthia_14.yaml resume results/adv_L2_norm/ >>./test_adv_unk_L2_norm.txt



#sd_test_model:  

CUDA_VISIBLE_DEVICES=0 python test_unk_C1_multiscale_log.py -cfg configs/deeplabv2_r101_tgt_synthia_self_distill_14.yaml resume results/sd_test_syn/ >>./sd_test_syn_multiscale_log.txt











