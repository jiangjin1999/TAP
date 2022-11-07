export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=22344 \
    model-trainer.py \
    --is_use_DDP True \
    --current_dataset AISHELL-1
