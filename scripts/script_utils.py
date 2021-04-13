import torch
import argparse

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

TRAIN_CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
VAL_CANCERS = ['ACC', 'CHOL', 'ESCA', 'LIHC', 'KICH', 'KIRC', 'OV', 'UCS', 'UCEC']

PARAMS = ['RENORMALIZE', 'TRAIN_FRAC', 'VAL_FRAC', 'BATCH_SIZE', 'WAIT_TIME', 'MAX_BATCHES', 'PIN_MEMORY', 'N_WORKERS', 'RANDOM_SEED',
          'TRAINING', 'LEARNING_RATE', 'WEIGHT_DECAY', 'DROPOUT', 'PATIENCE', 'FACTOR', 'N_EPOCHS', 'DISABLE_CUDA', 
          'OUT_DIM', 'MIN_TILES', 'NUM_TILES', 'UNIT', 'POOL', 'CANCERS', 'METADATA', 'STATE_DICT', 'VAL_STATS', 
          'VAL_CANCERS', 'TEST_VAL', 'HID_DIM', 'FREEZE', 'RES_DICT', 'RES_DICT_NEW', 'N_STEPS', 'N_TESTTRAIN', 'GRAD_ADAPT', 'ETA', 'N_CHOOSE']

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
    parser.add_argument('--max_batches', type=int, default=20, help='max number of batches per epoch (-1: include all)')
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

    # I/O parameters
    parser.add_argument('--output_size', type=int, default=1, help='model output dimension')
    parser.add_argument('--min_tiles', type=int, default=1, help='min number of tiles for patient to be included during sampling')
    parser.add_argument('--num_tiles', type=int, default=400, help='max number of tiles to retain per patient')
    parser.add_argument('--unit', type=str, default='tile', help='input unit, i.e., whether to train on tile or slide')
    parser.add_argument('--pool', type=str, default=None, help='pooling mechanism to use if input unit is slide')
    parser.add_argument('--cancers', nargs='*', default=TRAIN_CANCERS, help='list of cancers to include [in the train set]')
    parser.add_argument('--infile', type=str, default=METADATA_FILEPATH, help='file path to metadata dataframe')
    parser.add_argument('--outfile', type=str, default='/home/schao/temp.pt', help='file path to save the model state dict')
    parser.add_argument('--statsfile', type=str, default='/home/schao/temp.pkl', help='file path to save the per-epoch val stats')
    
    # task parameters
    parser.add_argument('--val_cancers', nargs='*', default=VAL_CANCERS, help='list of cancers to include in the val set')
    parser.add_argument('--test_val', default=False, action='store_true', help='whether to test model on val cancers')
    parser.add_argument('--hidden_size', type=int, default=512, help='feed forward hidden size')
    parser.add_argument('--freeze', default=False, action='store_true', help='whether to freeze the resnet layers')
    parser.add_argument('--resfile', type=str, default=None, help='path to pre-trained resnet')
    parser.add_argument('--resfile_new', type=str, default=None, help='path to newly-trained resnet, if freeze is false')
    parser.add_argument('--n_steps', type=int, default=1, help='number of gradient steps to take on val set')
    parser.add_argument('--n_testtrain', type=int, default=50, help='number of examples on which to train during test time')
    parser.add_argument('--grad_adapt', default=False, action='store_true', help='whether to grad adapt in non-meta-learn sits if testing on val cancers')
    
    # maml parameters
    parser.add_argument('--eta', type=float, default=0.01, help='global learning rate')
    parser.add_argument('--n_choose', type=int, default=5, help='number of tasks to sample during every training epoch')

    args = parser.parse_args()
    
    args.pool = POOL_KEY.get(args.pool, None)
    
    return args