import h5py
import numpy as np
import _init_paths
from viztool.landscape import Surface, Dir2D, Sampler
from viztool import projection as proj, scheduler

sur1 = Surface.load('output/imagenet/cls_mdeq_SMALL/surf_[-0.2,0.2,40]x[-0.2,0.2,40].h5')
sur2 = Surface.load('output/o/imagenet/cls_mdeq_SMALL/surf_[-0.2,0.2,40]x[-0.2,0.2,40].h5')

layers = sur1.layers.keys()

for layer in layers:
    mer = np.amax(np.stack((sur1.layers[layer],sur2.layers[layer]), axis=-1), axis=-1)
    assert mer.shape == sur1.layers[layer].shape
    sur1.layers[layer][:] = mer[:]
    sur2.layers[layer][:] = mer[:]
sur1.save('w')
sur2.save('w')


# inds, coords, inds_nums = scheduler.get_job_indices(*sur1.get_unplotted_indices('loss'), 0, 2)
# with h5py.File('output/imagenet/cls_mdeq_SMALL/job.h5', 'w-') as f:
#     f['inds'] = inds
#     f['coords'] = coords


# inds, coords, inds_nums = scheduler.get_job_indices(*sur1.get_unplotted_indices('loss'), 1, 2)
