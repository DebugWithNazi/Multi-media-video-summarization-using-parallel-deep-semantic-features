from __future__ import print_function
from src.msva_models import MSVA_Gen_auto
import glob
import cv2
import os
import os.path as osp
import argparse
import sys
import h5py
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from utils import Logger, read_json, write_json, save_checkpoint
import src.vsum_tools as  vsum_tools

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('-d', '--dataset', type=str, default='.\\datasets\\object_features\\dataset_custom_processed.h5',
                    help="path to h5 dataset (required)")

parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, default='custom', choices=['tvsum', 'summe','custom'],
                    help="evaluation metric ['tvsum', 'summe','custom']")
# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', default=True, action='store_true', help="use cpu device")
parser.add_argument('--evaluate', default=True, action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='.\\model_weights\\summe_random_non_overlap_0.5359.tar.pth',
                    help="path to resume file")
parser.add_argument('--verbose', default=True, action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', default=True, action='store_true', help="whether to save output results")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False







class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def getSmoothOutput(yArray, mastSize=5):
    maskedOut = []
    for i in range(len(yArray) - mastSize):
        maskedOut.append(np.mean(yArray[i:i + mastSize]))
    return maskedOut

def GenerateSummaryFromH5():
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    video_keys = list(dataset.keys())
    print("Initialize model")
    cmb = [1, 1, 1]
    feat_input = {"feature_size": 365, "L1_out": 365, "L2_out": 365, "L3_out": 512, "pred_out": 1, "apperture": 250,
                  "dropout1": 0.5, "att_dropout1": 0.5, "feature_size_1_3": 1024, "feature_size_4": 365}
    feat_input_obj = obj(feat_input)
    model = MSVA_Gen_auto(feat_input_obj, cmb)
    model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
    evaluate(model, dataset, video_keys, use_gpu)


def evaluate(model, dataset, video_keys, use_gpu):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(video_keys):
            print("generating summary for  '{}'".format(key_idx))
            seq1 = dataset[key]['features'][...]
            minShape = np.min([seq1.shape[0]])
            seq1 = cv2.resize(seq1, (seq1.shape[1], minShape), interpolation=cv2.INTER_AREA)
            seq2 =  dataset[key]['features_flow'][...]
            seq3 =  dataset[key]['features_rgb'][...]
            minShape = np.min([seq1.shape[0], seq2.shape[0], seq3.shape[0]])
            seq1 = cv2.resize(seq1, (seq1.shape[1], minShape), interpolation=cv2.INTER_AREA)
            seq2 = cv2.resize(seq2, (seq2.shape[1], minShape), interpolation=cv2.INTER_AREA)
            seq3 = cv2.resize(seq3, (seq3.shape[1], minShape), interpolation=cv2.INTER_AREA)
            seq_len = seq1.shape[1]
            seq1 = torch.from_numpy(seq1).unsqueeze(0)
            seq2 = torch.from_numpy(seq2).unsqueeze(0)
            seq3 = torch.from_numpy(seq3).unsqueeze(0)
            print("feature source 1 shape: ", seq1.shape)
            print("feature source 2 shape: ", seq2.shape)
            print("feature source 3 shape: ", seq3.shape)

            y, _ = model([seq1, seq2, seq3], seq_len)

            yArray = y.detach().numpy()[0]
            yArray = getSmoothOutput(yArray)

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            #user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(yArray, cps, num_frames, nfps, positions)

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=yArray)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/video_name', data=dataset[key]['video_name'])


                #h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
               # h5_res.create_dataset(key + '/fm', data=fm)

    if args.save_results: h5_res.close()
    print('All Files processed')




if __name__ == '__main__':
    GenerateSummaryFromH5()
