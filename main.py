
from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder, make_folder_simple

import glob
import os

def main(config):
    # For fast training
    cudnn.benchmark = True


    config.n_class = len(glob.glob(os.path.join(config.image_path, '*/')))
    print('number class:', config.n_class)
    # Data loader
    data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
                             config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder_simple(config.experiment_path)
    model_save_path = os.path.join(config.experiment_path, 'models')
    sample_path = os.path.join(config.experiment_path, 'samples')
    log_path = os.path.join(config.experiment_path, 'logs')
    attn_path = os.path.join(config.experiment_path, 'attn')

    make_folder_simple(model_save_path)
    make_folder_simple(sample_path)
    make_folder_simple(log_path)
    make_folder_simple(attn_path)


    print('config data_loader and build logs folder')

    if config.train:
        if config.model=='sagan':
            trainer = Trainer(data_loader.loader(), config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)