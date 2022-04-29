from solver.ddp_mix_solver import DDPMixSolver
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 50003 main.py >> train.log 2>&1 &


def init_distributed_mode(rank, ws):
    dist.init_process_group(backend='nccl', init_method='tcp://10.15.198.46:3311',
                            world_size=ws, rank=rank)
    rank = dist.get_rank()
    print(f"rank = {rank} is initialized")
    torch.cuda.set_device(rank)


def main(rank, ws, cfg_path):
    init_distributed_mode(rank, ws)
    processor = DDPMixSolver(cfg_path=cfg_path)
    processor.run()


if __name__ == '__main__':
    world_size = 2
    cfg_path = "config/mine.yaml"
    mp.spawn(main, nprocs=world_size, args=(world_size, cfg_path))



    # processor = DDPMixSolver(cfg_path="config/mine.yaml")
    # processor.run()
