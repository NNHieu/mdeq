# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.cls_function import validate
from utils.modelsummary import get_model_summary

import socket
import h5py
from viztool.landscape import Surface, Dir2D, Sampler
from viztool import projection as proj, scheduler
from utils.utils import create_logger
# from utils import mpi4pytorch as mpi, mpilogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--resolution', type=int, nargs=2, required=True)
    parser.add_argument('--rect', type=float, nargs=4, required=True)
    parser.add_argument('--rank', type=int, default=-1, required=True)
    args = parser.parse_args()
    update_config(config, args)

    return args

def name_surface_file(rect, res, dir_file):
    # # skip if surf_file is specified in args
    # if args.surf_file:
    #     return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file
    xmin, ymin, xmax, ymax = rect
    xnum, ynum = res
    # resolution
    surf_file += '_[%s,%s,%d]' % (str(xmin), str(xmax), int(xnum))
    surf_file += 'x[%s,%s,%d]' % (str(ymin), str(ymax), int(ynum))

    # dataloder parameters
    # if raw_data: # without data normalization
    #     surf_file += '_rawdata'
    # if data_split > 1:
    #     surf_file += '_datasplit=' + str(data_split) + '_splitidx=' + str(split_idx)

    return surf_file + ".h5"

def create_surfile(model, layers, dir_file, surf_file, rect, resolution, logger):
    if not os.path.exists(dir_file):
        logger.info('Create dir file at {}'.format(dir_file))
        dir2d = Dir2D(model=model)
        with h5py.File(dir_file, 'w') as f:
            dir2d.save(f)
    
    if not os.path.exists(surf_file):
        logger.info('Create surface file at {}'.format(surf_file))
        surface = Surface(dir_file, rect, resolution, surf_file, {})
        surface.add_layer(*layers)
        surface.save()
    
    return surf_file

def get_loader(n_gpu):
    # Data loading code
    valdir = os.path.join(config.DATASET.ROOT,
                          config.DATASET.TEST_SET)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        persistent_workers=True,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*n_gpu,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    return valid_loader

def mpi_run(args):
    
    
    #--------------------------------------------------------------------------
    # MPI setup
    #--------------------------------------------------------------------------
    # if args.mpi:
        # comm = mpi.setup_MPI()
        # rank, nproc = comm.Get_rank(), comm.Get_size()
    # else:
    print('Non mpi')
    comm, rank, nproc = None, 0, 1
    if args.rank >= 0:
    #     rank = args.rank
        nproc = 2
    #     print('set rank', rank)
    #--------------------------------------------------------------------------
    # Logger setup
    #--------------------------------------------------------------------------

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid', mpi=args.mpi)
    if rank == 0:
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

    if rank <= 1:
        #--------------------------------------------------------------------------
        # Cuda setup
        #--------------------------------------------------------------------------
        
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(args.rank % gpu_count)
        device = f"cuda:{args.rank % gpu_count}"
        logger.info('Rank %d use GPU %d of %d GPUs on %s' %
                (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))
        
        #--------------------------------------------------------------------------
        # Prepair model, surface
        #--------------------------------------------------------------------------
        
        model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
        dump_input = torch.rand(
            (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
        )
        # logger.info(get_model_summary(model, dump_input))

        if config.TEST.MODEL_FILE:
            logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        else:
            model_state_file = os.path.join(final_output_dir,
                                            'final_state.pth.tar')
            logger.info('=> loading model from {}'.format(model_state_file))
            model.load_state_dict(torch.load(model_state_file))

        dir_file = os.path.join(final_output_dir, 'dir.h5')
        surf_file = os.path.join(final_output_dir, name_surface_file(args.rect, args.resolution, 'surf'))
        layers = ('loss', 'err1', 'err5')
        if rank == 0:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            create_surfile(model, layers, dir_file, surf_file, args.rect, args.resolution, logger)
        # wait until master has setup the direction file and surface file
    # mpi.barrier(comm)
    if rank <= 1:
        surface = Surface.load(surf_file)
        dir2d = surface.dirs
        similarity = proj.cal_angle(proj.nplist_to_tensor(dir2d[0]), proj.nplist_to_tensor(dir2d[1]))
        logger.info('cosine similarity between x-axis and y-axis: %f' % similarity)
        del similarity
        #--------------------------------------------------------------------------
        # Prepair Data, Cuda
        #--------------------------------------------------------------------------
        # gpus = list(range(gpu_count))
        # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        model.to(device)
        # define loss function (criterion) and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        criterion.to(device)
        
        valid_loader = get_loader(gpu_count)
        print(len(os.sched_getaffinity(0)))
        def evaluation(model):
            # evaluate on validation set
            loss, err1, err5 = validate(config, valid_loader, model, criterion, None, -1,
                    final_output_dir, tb_log_dir, None, topk=(1,5), device=device)
            return loss, err1, err5

        sampler = Sampler(model, surface, layers, None, comm=None, rank=rank, logger=logger)
        sampler.prepair()
        inds, coords, inds_nums = scheduler.get_job_indices(*surface.get_unplotted_indices('loss'), rank if args.mpi or args.rank < 0 else args.rank, nproc)
        surface.open('r+')
        sampler.run(evaluation, inds, coords, inds_nums)
        surface.close()
    # mpi.barrier(comm)

    

    

def main():
    args = parse_args()
    

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    mpi_run(args)


if __name__ == '__main__':
    main()
