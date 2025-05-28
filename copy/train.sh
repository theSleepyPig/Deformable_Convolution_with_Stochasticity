CUDA_VISIBLE_DEVICES=0,1,3 
torchrun 
--nproc_per_node=3 train_vit_pretrain_ddp.py     \
--optimizer sgd     \
--network earlyVit   \
  --dataset cifar10    \ 
  --lr_max 0.1   \  
  --weight_decay 5e-4  \  
  --lr_schedule multistep  \  
    --batch_size 128  \   
  --epochs 200    \ 
  --fix