import torch
import argparse

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

TRAIN_CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
VAL_CANCERS = ['ACC', 'CHOL', 'ESCA', 'LIHC', 'KICH', 'KIRC', 'OV', 'UCS', 'UCEC']

PARAMS = ['RENORMALIZE', 'TRAIN_FRAC', 'VAL_FRAC', 'BATCH_SIZE', 'WAIT_TIME', 'MAX_BATCHES', 'PIN_MEMORY', 'N_WORKERS', 'RANDOM_SEED',
          'TRAINING', 'LEARNING_RATE', 'WEIGHT_DECAY', 'DROPOUT', 'PATIENCE', 'FACTOR', 'N_EPOCHS', 'DISABLE_CUDA', 'DEVICE',
          'OUT_DIM', 'MIN_TILES', 'NUM_TILES', 'LABEL', 'UNIT', 'POOL', 'CANCERS', 'METADATA', 'STATE_DICT', 'VAL_STATS', 
          'VAL_CANCERS', 'TEST_VAL', 'HID_DIM', 'FREEZE', 'PRETRAINED', 'RES_DICT', 'RES_DICT_NEW', 'GRAD_ADAPT', 
          'ETA', 'N_CHOOSE', 'N_STEPS', 'N_TESTTRAIN', 'N_TESTTEST', 'N_REPLICATES', 'TEST_BATCH_SIZE', 'RANDOMIZE', 'BRIGHTNESS', 'RESIZE', 'STEPS']

POOL_KEY = {
    'max': torch.max,
    'mean': torch.mean,
    'lse': torch.logsumexp
}

def parse_args():
    parser = argparse.ArgumentParser(description='WGD classifier')

    # data parameters
    parser.add_argument('--renormalize', default=False, action='store_true', help='whether to recompute mean and std of train set')
    parser.add_argument('--train_frac', type=float, default=0.8, help='fraction of examples allocated to the train set')
    parser.add_argument('--val_frac', type=float, default=0.2, help='fraction of examples allocated to the val set')
    parser.add_argument('--batch_size', type=int, default=200, help='number of examples per batch')
    parser.add_argument('--wait_time', type=int, default=1, help='number of batches collected before backward pass')
    parser.add_argument('--max_batches', nargs='*', type=int, default=[-1, -1], help='max number of batches during train, val per epoch (-1: all)')
    parser.add_argument('--pin_memory', default=False, action='store_true', help='whether to pin memory during data loading')
    parser.add_argument('--n_workers', type=int, default=12, help='number of workers to use during data loading')
    parser.add_argument('--random_seed', type=int, default=31321, help='random seed of the dataset and data filter')
    
    # learning parameters
    parser.add_argument('--training', default=False, action='store_true', help='whether to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='[local] learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight assigned to L2 regularizer')
    parser.add_argument('--dropout', type=float, default=0.0, help='feed forward dropout rate')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs with no improvement before invoking scheduler, model reloading')
    parser.add_argument('--factor', type=float, default=0.1, help='factor by which to reduce learning rate during scheduling')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train the model')
    parser.add_argument('--disable_cuda', default=False, action='store_true', help='whether or not to use GPU')
    parser.add_argument('--device', type=str, default='0', help='CUDA device to use if use GPU')

    # I/O parameters
    parser.add_argument('--output_size', type=int, default=1, help='model output dimension')
    parser.add_argument('--min_tiles', type=int, default=1, help='min number of tiles for patient to be included during sampling')
    parser.add_argument('--num_tiles', type=int, default=400, help='number of tiles to keep (tile) or sample (slide) per patient')
    parser.add_argument('--label', type=str, default='WGD', help='label on which to perform classification task')
    parser.add_argument('--unit', type=str, default='tile', help='input unit, i.e., whether to train on tile or slide')
    parser.add_argument('--pool', type=str, default=None, help='pooling mechanism to use if input unit is slide')
    parser.add_argument('--cancers', nargs='*', type=str, default=TRAIN_CANCERS, help='list of cancers to include [in the train set]')
    parser.add_argument('--infile', type=str, default=METADATA_FILEPATH, help='file path to metadata dataframe')
    parser.add_argument('--outfile', type=str, default='/home/schao/temp.pt', help='file path to save the model state dict')
    parser.add_argument('--statsfile', type=str, default='/home/schao/temp.pkl', help='file path to save the per-epoch val stats')
    
    # task parameters
    parser.add_argument('--val_cancers', nargs='*', type=str, default=VAL_CANCERS, help='list of cancers to include in the val set')
    parser.add_argument('--test_val', default=False, action='store_true', help='whether to test non-meta-learned model on val cancers')
    parser.add_argument('--skip', type=int, default=0, help='number of metaval or metatest loaders to skip')
    parser.add_argument('--hidden_size', type=int, default=512, help='feed forward hidden size')
    parser.add_argument('--freeze', default=False, action='store_true', help='whether to freeze the resnet layers')
    parser.add_argument('--pretrained', default=False, action='store_true', help='whether to load the ImageNet-pretrained resnet')
    parser.add_argument('--resfile', type=str, default=None, help='path to resnet')
    parser.add_argument('--resfile_new', type=str, default=None, help='path to newly-trained resnet, if freeze is false')
    parser.add_argument('--grad_adapt', default=False, action='store_true', help='whether to grad adapt non-meta-learned model during test')
    
    # meta-learning parameters
    parser.add_argument('--eta', type=float, default=0.01, help='global learning rate')
    parser.add_argument('--n_choose', type=int, default=5, help='number of tasks to sample during every training epoch')
    parser.add_argument('--n_steps', type=int, default=1, help='number of gradient steps to take on meta-test train set')
    parser.add_argument('--n_testtrain', type=int, default=0, help='number of examples on which to train during meta-test time or train time')
    parser.add_argument('--n_testtest', type=int, default=0, help='number of examples on which to test during meta-test time or test time')
    parser.add_argument('--n_replicates', type=int, default=1, help='number of replicates for metaval and metatest')    
    parser.add_argument('--test_batch_size', type=int, default=4, help='number of examples per meta-test test batch')
    parser.add_argument('--randomize', default=False, action='store_true', help='whether to randomize the train size during meta-train/-test')
    parser.add_argument('--adjust_brightness', type=float, default=None, help='desired brightness (<1 darker, >1 brighter) on meta-test set')
    parser.add_argument('--resize', type=int, default=None, help='desired image size to which to interpolate on meta-test set')
    parser.add_argument('--steps', type=int, default=None, help='desired number of grad steps up to which to test on meta-test set (overrides n_steps)')
    
    args = parser.parse_args()
    
    args.pool = POOL_KEY.get(args.pool, None)
    
    return args