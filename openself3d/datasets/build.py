import platform

from mmcv.utils import Registry, build_from_cfg

if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASOURCES = Registry('datasource')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')



def build_datasource(cfg, DATASOURCES):
    pass


def build_dataset(cfg, default_args=None):
    pass


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     **kwargs):
    pass

def worker_init_fn(worker_id, num_worker, rank, seed):
    # The seed of each worker equels to 
    # num_worker * rank + worker_id +user_seed
    worker_seed = num_worker * rank + worker_id + seed
    
    