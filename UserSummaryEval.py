import sys
import torch

from src import vsum_tools
from src.msva_models import MSVA_Gen_auto
import numpy as np
# from src.sys_utils import plotData
import h5py
import os
import glob
import cv2
from matplotlib import pyplot as plt


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


def getParameter(param):
    for i in range(len(sys.argv)):
        if (sys.argv[i] == param):
            return sys.argv[i + 1]


def getVidIndexFromObjectFeatures(video_name, dataset):
    allFiles = glob.glob(
        "datasets" + os.sep + "kinetic_features" + os.sep + dataset + os.sep + "FLOW" + os.sep + "features" + os.sep + "*")
    videoDict = {}
    for i in range(len(allFiles)):
        fname = allFiles[i].split(os.sep)[-1].split(".")[0]
        videoDict[fname] = "video_" + str(i + 1)
    return videoDict

dataset = "summe"
video_name = "Air_Force_One"
model_weight = "model_weights/summe_random_non_overlap_0.5359.tar.pth"
#dataset = getParameter("-dataset")
#video_name = getParameter("-video_name")
#model_weight = getParameter("-model_weight")

videoDict = getVidIndexFromObjectFeatures(video_name, dataset)
saveImg = True
dpi = 100

if (video_name not in videoDict):
    print("video name not found !!!")
else:
    #print("video name found and corresponding index: ", videoDict[video_name])
    video_index = videoDict[video_name]
    cmb = [1, 1, 1]
    feat_input = {"feature_size": 365, "L1_out": 365, "L2_out": 365, "L3_out": 512, "pred_out": 1, "apperture": 250,
                  "dropout1": 0.5, "att_dropout1": 0.5, "feature_size_1_3": 1024, "feature_size_4": 365}
    feat_input_obj = obj(feat_input)
    model = MSVA_Gen_auto(feat_input_obj, cmb)
    model.load_state_dict(torch.load(model_weight, map_location=lambda storage, loc: storage))

    datasetsH5 = h5py.File("datasets" + os.sep + "object_features" + os.sep + "eccv16_dataset_summe_google_pool5.h5",
                           'r')
    seq1 = datasetsH5[videoDict[video_name]]['features'][...]
    seq2 = np.load(
        "datasets" + os.sep + "kinetic_features" + os.sep + "summe" + os.sep + "FLOW" + os.sep + "features" + os.sep + video_name + ".npy")  # you can provide features from another video
    seq3 = np.load(
        "datasets" + os.sep + "kinetic_features" + os.sep + "summe" + os.sep + "RGB" + os.sep + "features" + os.sep + video_name + ".npy")
    minShape = np.min([seq1.shape[0], seq2.shape[0], seq3.shape[0]])
    seq1 = cv2.resize(seq1, (seq1.shape[1], minShape), interpolation=cv2.INTER_AREA)
    seq2 = cv2.resize(seq2, (seq2.shape[1], minShape), interpolation=cv2.INTER_AREA)
    seq3 = cv2.resize(seq3, (seq3.shape[1], minShape), interpolation=cv2.INTER_AREA)
    seq_len = seq1.shape[1]
    seq1 = torch.from_numpy(seq1).unsqueeze(0)
    seq2 = torch.from_numpy(seq2).unsqueeze(0)
    seq3 = torch.from_numpy(seq3).unsqueeze(0)

    y, _ = model([seq1, seq2, seq3], seq_len)
    yArray = y.detach().numpy()[0]
    yArray = getSmoothOutput(yArray)
    cps = datasetsH5[videoDict[video_name]]['change_points'][...]
    num_frames = datasetsH5[videoDict[video_name]]['n_frames'][()]
    nfps = datasetsH5[videoDict[video_name]]['n_frame_per_seg'][...].tolist()
    positions = datasetsH5[videoDict[video_name]]['picks'][...]
    user_summary = datasetsH5[videoDict[video_name]]['user_summary'][...]

    machine_summary = vsum_tools.generate_summary(yArray, cps, num_frames, nfps, positions)
    fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'avg')
    print('Average F1 score : '+str(fm))

    fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'max')
    print('Max F1 score : '+str(fm))

