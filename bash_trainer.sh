export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=22345 \
    model-trainer.py 
