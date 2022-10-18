sleep 7h
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
<<<<<<< HEAD
    --nproc_per_node=8 \
=======
    --nproc_per_node=4 \
>>>>>>> 40d48d0c0ba4df4e4a65a7f81556b1e0554a0d33
    model-trainer.py 