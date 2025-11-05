import torch
import os
from lib import utils as ut
import torch.distributed as dist

def log(output_dir, acc, ep):
    with open(output_dir / 'LOG.log', 'a') as f:
        f.write(str(ep) + ' ' + str(acc) + '\n')
        
def log70(output_dir, acc, ep):
    with open(output_dir / 'LOG_70.log', 'a') as f:
        f.write(str(ep) + ' **********acc70: ' + str(acc) + '\n')
        
def log50(output_dir, acc, ep):
    with open(output_dir / 'LOG_50.log', 'a') as f:
        f.write(str(ep) + ' #############################acc50: ' + str(acc) + '\n')

@torch.no_grad()
def save(checkpoins, acc, ep, output_dir):
    checkpoint_path = output_dir / 'best_checkpoins.pt'
    ut.save_on_master(checkpoins, checkpoint_path)
    # with open(output_dir / 'best_acc.log', 'w') as f:
    #     f.write(str(ep)+': '+str(acc))

##############################TODO:
##############################TODO:

# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print

# def init_distributed_mode(args):

#     if args.dist_on_itp:
#         args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
#         args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
#         args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
#         args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'],
#                                          os.environ['MASTER_PORT'])
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['RANK'] = str(args.rank)
#         os.environ['WORLD_SIZE'] = str(args.world_size)
#         # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
#     elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()

#         os.environ['RANK'] = str(args.rank)
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['WORLD_SIZE'] = str(args.world_size)
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}, gpu {}'.format(
#         args.rank, args.dist_url, args.gpu),
#           flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend,
#                                          init_method=args.dist_url,
#                                          world_size=args.world_size,
#                                          rank=args.rank)
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)

# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True

# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()

##############################TODO:
##############################TODO: