"""
This training script can run on a single GPU (cuda or mps) and also on multiple
GPUs with distributed data parallel (ddp).

To run on single GPU:
$ python train.py --batch_size=32 --compile=False

To run on multiple GPUs:
$ torchrun --standalone --nproc_per_node=4 train.py

"""