import os

from .coco import CocoDataset
from .custom_transform import Normalizer, Augmenter, Resizer, ToTensor
from .utils import collater
from .visdrone import VisDroneDataset

from torchvision import transforms
from torch.utils.data import DataLoader

def DatasetInit(opt, params):
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

    if opt.project == 'coco':
        training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                                   transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                                 Augmenter(),
                                                                 Resizer(input_sizes[opt.compound_coef])]))
        training_generator = DataLoader(training_set, **training_params)

        val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                              transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                            Resizer(input_sizes[opt.compound_coef])]))
        val_generator = DataLoader(val_set, **val_params)

    elif opt.project == 'visdrone':
        training_set = VisDroneDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                                       transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                                     Augmenter(), Resizer(input_sizes[opt.compound_coef]),
                                                                     ToTensor()]))
        training_generator = DataLoader(training_set, **training_params)

        val_set = VisDroneDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                              transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                            Resizer(input_sizes[opt.compound_coef]), ToTensor()]))
        val_generator = DataLoader(val_set, **val_params)

    else:
        raise NotImplementedError

    return training_generator, val_generator
