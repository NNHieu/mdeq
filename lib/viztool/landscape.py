import torch
from torch import Tensor

from logging import debug
import os
import sys
from warnings import resetwarnings
from torch._C import device

from torch.functional import Tensor
from .utils import read_list, write_list
import h5py

# import torch.distributed as dist

import numpy as np
from . import projection as proj, scheduler
import torch.multiprocessing as mp
import time
from functools import partial
from utils import mpi4pytorch as mpi

def get_weights_as_list(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]

class Direction:
    ################################################################################
    #                        Normalization Functions
    ################################################################################
    @staticmethod
    def _normalize(direction: Tensor, weights: Tensor, norm='filter'):
        """
            Rescale the direction so that it has similar norm as their corresponding
            model in different levels.

            Args:
            direction (tensor): a variables of the random direction for one layer
            weights (tensor): a variable of the original model for one layer
            norm: normalization method, 'filter' | 'layer' | 'weight'
        """
        if norm == 'filter':
            # Rescale the filters (weights in group) in 'direction' so that each
            # filter has the same norm as its corresponding filter in 'weights'.
            assert direction.dim() == 3 and weights.dim() == 3
            sc = weights.norm(dim=(-1, -2), keepdim=True)/(direction.norm(dim=(-1, -2), keepdim=True) + 1e-10)
            direction.mul_(sc)
        elif norm == 'layer':
            # Rescale the layer variables in the direction so that each layer has
            # the same norm as the layer variables in weights.
            direction.mul_(weights.norm()/direction.norm())
        elif norm == 'weight':
            # Rescale the entries in the direction so that each entry has the same
            # scale as the corresponding weight.
            direction.mul_(weights)
        elif norm == 'dfilter':
            # Rescale the entries in the direction so that each filter direction
            # has the unit norm.
            dnorm = direction.view(direction.size(0), -1).norm(dim=-1).view(direction.size())
            direction.div_(dnorm + 1e-10)
        elif norm == 'dlayer':
            # Rescale the entries in the direction so that each layer direction has
            # the unit norm.
            direction.div_(direction.norm())

    @staticmethod
    def normalize_for_weights(direction, weights, norm='filter', ignore='biasbn'):
        """
            The normalization scales the direction entries according to the entries of weights.
        """
        assert(len(direction) == len(weights))
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn': d.fill_(0) # ignore directions for weights with 1 dimension
                else:                  d.copy_(w) # keep directions for weights/bias that are only 1 per node
            else:
                Direction._normalize(d, w, norm)
    
    @staticmethod
    def create_random_direction(params, ignore='biasbn', norm=True, norm_type='filter'):
        """
            Setup a random (normalized) direction with the same dimension as
            the weights or states.

            Args:
            net: the given trained model
            dir_type: 'weights' or 'states', type of directions.
            ignore: 'biasbn', ignore biases and BN parameters.
            norm: direction normalization method, including
                    'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

            Returns:
            direction: a random direction with the same dimension as weights or states.
        """

        # random direction
        weights_data = [p.data for p in params] # a list of parameters.
        direction = [torch.randn(w.size()) for w in params]
        if norm:
            Direction.normalize_for_weights(direction, weights_data, norm_type, ignore)
        return direction
    
    @staticmethod
    def set_weights(net, weights, directions=None, step=None):
        """
            Overwrite the network's weights with a specified list of tensors
            or change weights along directions with a step size.
        """
        if directions is None:
            # You cannot specify a step length without a direction.
            for (p, w) in zip(net.parameters(), weights):
                p.data.copy_(w.type(type(p.data)))
        else:
            assert step is not None, 'If a direction is specified then step must be specified as well'
            if len(directions) == 2:
                dx = directions[0]
                dy = directions[1]
                # self.logger.info(dx)
                # self.logger.info(len(dx), len(dy))
                changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
            else:
                changes = [d*step for d in directions[0]]
            # self.logger.info('change norm', torch.norm(proj.tensorlist_to_tensor(changes)))
            # self.logger.info(torch.norm(proj.tensorlist_to_tensor(changes)))

            for (p, w, d) in zip(net.parameters(), weights, changes):
                s = w + d.to(w.device)
                p.data = s
            
    @staticmethod
    def save(direction, h5_file, name):
        # Create the plotting directions
        write_list(h5_file, name, direction)

    def load(h5_file, name):
        # Create the plotting directions
        return read_list(h5_file, name)
    
    @staticmethod
    def to_tensor(dir, **kwargs):
        return [torch.tensor(arr, **kwargs) for arr in dir]

class Dir2D(object):
    def __init__(self, model=None, dirs = None, mode='random') -> None:
        super().__init__()
        if model is not None:
            weights = get_weights_as_list(model) # List representation
            dir0 = Direction.create_random_direction(weights, norm_type='layer')
            dir1 = Direction.create_random_direction(weights, norm_type='layer')
            self._dir = (dir0, dir1)
            # Todo: Assert these dir are othorgonal
        elif dirs is not None and len(dirs) == 2:
            self._dir = dirs
        

    def save(self, h5file):
        Direction.save(self[0], h5file, 'xdir')
        Direction.save(self[1], h5file, 'ydir')


    @classmethod
    def load(cls, h5file):
        dir0 = Direction.load(h5file, 'xdir')
        dir1 = Direction.load(h5file, 'ydir')
        return cls(dirs=(dir0, dir1))

    def __getitem__(self, dir_index):
        return self._dir[dir_index]
    
    def __len__(self):
        return len(self._dir)
    
    def to_tensor(self, **kwargs):
        self.tensors = (Direction.to_tensor(self._dir[0], **kwargs), 
                            Direction.to_tensor(self._dir[1], **kwargs))
    
    def tensor(self, dir_index):
        return self.tensors[dir_index]
            

class Surface:
    def __init__(self, path_dir2d, rect, resolution, path, layers) -> None:
        with h5py.File(path_dir2d) as f:
            self.dirs = Dir2D.load(f)
        self.dir_path = path_dir2d
        xmin, ymin, xmax, ymax = rect
        xnum, ynum = int(resolution[0]), int(resolution[1])
        self.xcoord = np.linspace(xmin, xmax, num=xnum)
        self.ycoord = np.linspace(ymin, ymax, num=ynum)
        self.shape = (xnum, ynum)
        self.path = path
        self.h5_file = None
        self.layers = layers

    def add_layer(self, *names, value=-1):
        for name in names:
            self.layers[name] = np.ones(self.shape)*value

    def mesh(self):
        return np.meshgrid(self.xcoord, self.ycoord)

    def save(self, mode='w-'):
        f = h5py.File(self.path, mode) # Create file, fail if exists
        f.attrs['dir_path'] = self.dir_path
        f['xcoord'] = self.xcoord 
        f['ycoord'] = self.xcoord
        layer_grp = f.create_group('layers')
        for name, values in self.layers.items():
            layer_grp.create_dataset(name, data=values)
        f.close()
    
    @classmethod
    def load(cls, path):
        f = h5py.File(path, 'r')
        direction_path = f.attrs['dir_path']
        xcoord = f['xcoord'][:]
        ycoord = f['ycoord'][:]
        layer_grp = f['layers']
        layers = {}
        for name, values in layer_grp.items():
            layers[name] = values[:]
        obj = cls(direction_path, (0, 0, 0, 0), (0, 0), path, layers)
        obj.xcoord = xcoord
        obj.ycoord = ycoord
        f.close()
        return obj
    
    def get_unplotted_indices(self, layer):
        """
        Args:
        layer: layer name, with value -1 when the value is not yet calculated.

        Returns:
        - a list of indices into vals for points that have not yet been calculated.
        - a list of corresponding coordinates, with one x/y coordinate per row.
        """

        # Create a list of indices into the vectorizes vals
        vals = self.layers[layer]
        inds = np.array(range(vals.size))

        # Select the indices of the un-recorded entries, assuming un-recorded entries
        # will be smaller than zero. In case some vals (other than loss values) are
        # negative and those indexces will be selected again and calcualted over and over.
        inds = inds[vals.ravel() <= 0]

        # Make lists containing the x- and y-coodinates of the points to be plotted
        # If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
        xcoord_mesh, ycoord_mesh = self.mesh()
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1,s2]

    def open(self, mode):
        self.h5_file = h5py.File(self.path, mode)
    
    def flush(self):
        assert self.h5_file, 'Have yet open'
        self.h5_file.flush()

    def close(self):
        assert self.h5_file, 'Have yet open'
        self.h5_file.close()

class Sampler:
    def __init__(self, model, surface, layer_names, device, comm=None, rank=-1, logger=None) -> None:
        self.model = model
        self.surface = surface
        self.rank = rank
        self.device = device
        self.layer_names = layer_names
        self.comm = comm
        self.logger = logger

    def prepair(self):
        # if rank == 0: self.surface.open('r+')
        self.surface.dirs.to_tensor()
        # Generate a list of indices of 'losses' that need to be filled in.
        # The coordinates of each unfilled index (with respect to the direction vectors
        # stored in 'd') are stored in 'coords'.
        # inds, coords, inds_nums = scheduler.get_job_indices(*surface.get_unplotted_indices(loss_key), rank, size)
        self.layers = [self.surface.layers[name] for name in self.layer_names]
        self.layers_fl = [layer.ravel() for layer in self.layers]
        model = self.model
        model.eval()
        # model.to(self.device)
    
    def reduce(self):
        # Send updated plot data to the master node
        if self.rank < 0: return 0
        syc_start = time.time()
        for layer in self.layers_fl:
            # dist.reduce(layer, 0, op=dist.ReduceOp.MAX)
            mpi.reduce_max(self.comm, layer)
        syc_time = time.time() - syc_start
        return syc_time
        
    
    def write(self):
        # Only the master node writes to the file - this avoids write conflicts
        if self.rank <= 0:
            for name, layer in zip(self.layer_names, self.layers):
                self.surface.h5_file['layers'][name][:] = layer
            self.surface.flush()

    def run(self, evaluation, inds, coords, inds_nums):
        """
            Calculate the loss values and accuracies of modified models in parallel
            using MPI reduce.
        """
        # dirs_tensor = (proj.tensorlist_to_tensor(directions[0]), proj.tensorlist_to_tensor(directions[1]))
        self.logger.info('Computing %d values for rank %d'% (len(inds), self.rank))
        start_time = time.time()
        total_sync = 0.0
        with torch.no_grad():
            model = self.model
            weights = [torch.clone(p) for p in model.parameters()]
            # Loop over all uncalculated loss values
            for count, ind in enumerate(inds):
                # Get the coordinates of the loss value being calculated
                coord = coords[count]
                Direction.set_weights(model, weights, self.surface.dirs.tensors, coord)
                # Record the time to compute the loss value
                loss_start = time.time()
                values = evaluation(model)
                loss_compute_time = time.time() - loss_start
                # Record the result in the local array
                for i, val in enumerate(values):
                    self.layers_fl[i][ind] = val

                syc_time = self.reduce()
                total_sync += syc_time
                self.write()

                log_values = '\t'.join(['{}={:.3f}'.format(name, val) for name, val in zip(self.layer_names, values)])
                self.logger.info('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s \ttime=%.2f \tsync=%.2f' % (
                        self.rank, count, len(inds), 100.0 * count/len(inds), str(coord), log_values, loss_compute_time, syc_time))

            # This is only needed to make MPI run smoothly. If this process has less work than
            # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
            for i in range(max(inds_nums) - len(inds)):
                self.reduce()

        total_time = time.time() - start_time
        self.logger.info('Rank %d done!  Total time: %.2f Sync: %.2f' % (self.rank, total_time, total_sync))
    
# def main():
#     model = None
#     surface = Surface.load(path)
#     loss_key = ('loss', 'acc')
#     inds, coords, inds_nums = scheduler.get_job_indices(*surface.get_unplotted_indices('loss'), 0, 1)
#     surface.open('r+')
#     sampler = Sampler(model, surface, loss_key,'gpu:0', 0)
#     sampler.prepair()
#     sampler.run(evaluation, inds, coords, inds_nums)
#     surface.close()
