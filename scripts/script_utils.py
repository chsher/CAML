import argparse

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

TRAIN_CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
VAL_CANCERS = ['ACC', 'CHOL', 'ESCA', 'LIHC', 'KICH', 'KIRC', 'OV', 'UCS', 'UCEC']

def parse_args():
    parser = argparse.ArgumentParser(description='WGD classifier')

    parser.add_argument('--train_frac', type=float, default=0.8, help='fraction of examples allocated to the train set')
    parser.add_argument('--val_frac', type=float, default=0.2, help='fraction of examples allocated to the val set')
    parser.add_argument('--batch_size', type=int, default=200, help='number of examples per batch')
    parser.add_argument('--wait_time', type=int, default=1, help='number of batches before backward pass')
    parser.add_argument('--max_batches', type=int, default=20, help='max number of batches per epoch (-1: include all)')
    parser.add_argument('--pin_memory', default=False, action='store_true', help='whether to pin memory during data loading')
    parser.add_argument('--n_workers', type=int, default=12, help='number of workers to use during data loading')

    parser.add_argument('--output_size', type=int, default=1, help='model output dimension')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='local learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight assigned to L2 regularization')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs with no improvement before invoking the scheduler, model reloading')
    parser.add_argument('--factor', type=float, default=0.1, help='factor by which to reduce learning rate during scheduling')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train the model')
    parser.add_argument('--disable_cuda', default=False, action='store_true', help='whether or not to use GPU')

    parser.add_argument('--num_tiles', type=int, default=400, help='number of tiles to include per slide')
    parser.add_argument('--unit', type=str, default='tile', help='input unit, i.e., whether to train on tile or slide')
    parser.add_argument('--cancers', nargs='*', default=TRAIN_CANCERS, help='list of cancers to include')
    parser.add_argument('--infile', type=str, default=METADATA_FILEPATH, help='file path to metadata dataframe')
    parser.add_argument('--outfile', type=str, default='temp.pt', help='file path to save the model state dict')
    parser.add_argument('--statsfile', type=str, default='temp.pkl', help='file path to save the per-epoch val stats')
    parser.add_argument('--training', default=False, action='store_true', help='whether to train the model')

    # -- new params --
    parser.add_argument('--val_cancers', nargs='*', default=VAL_CANCERS, help='list of cancers to include in the val set')
    parser.add_argument('--hidden_size', type=int, default=512, help='feed forward hidden size')
    parser.add_argument('--resfile', type=str, default=None, help='path to pre-trained resnet')
    parser.add_argument('--dropout', type=float, default=0.0, help='feed forward dropout')
    parser.add_argument('--n_steps', type=int, default=1, help='number of gradient steps to take on val set')
    parser.add_argument('--n_testtrain', type=int, default=50, help='number of examples on which to train during test')
    # ----------------
    parser.add_argument('--eta', type=float, default=0.01, help='global learning rate')
    parser.add_argument('--n_choose', type=int, default=5, help='number of tasks to sample during every training epoch')
    # ----------------

    args = parser.parse_args()
    
    return args