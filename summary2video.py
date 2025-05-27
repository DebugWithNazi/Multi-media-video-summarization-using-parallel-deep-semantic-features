import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse
from tkinter import *
from tkvideo import *

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='.\\log\\result.h5', help="path to h5 result file")
parser.add_argument('-i', '--idx', type=int, default=0, help="which key to choose")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")
parser.add_argument('--save-dir', type=str, default='log', help="directory to save")
args = parser.parse_args()

def frm2video(frm_dir, summary, vid_writer,summaryProgress,GenerateVideo):
    summaryProgress["value"] = 0
    GenerateVideo.update()
    for idx, val in enumerate(summary):
        if val == 1:
            # here frame name starts with '000001.jpg'
            # change according to your need
            #frm_name = str(idx+1).zfill(6) + '.jpg'
            #work arround for video
            frm_name = str(idx+1) + '.jpg'
            print(frm_name)
            frm_path = osp.join(frm_dir, frm_name)
            frm = cv2.imread(frm_path)
            frm = cv2.resize(frm, (args.width, args.height))
            vid_writer.write(frm)
        summaryProgress["value"] = (idx/len(summary))*100
        GenerateVideo.update()

def generateVideoSummary(videolabel,summaryProgress,GenerateVideo):
    if not osp.exists(args.save_dir):
        os.mkdir(args.save_dir)
    h5_res = h5py.File(args.path, 'r')
    key = list(h5_res.keys())[args.idx]
    summary = h5_res[key]['machine_summary'][...]
    name = h5_res[key]['video_name'][()].decode("utf-8")
    vid_writer = cv2.VideoWriter(
        osp.join(args.save_dir, name+'.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        (args.width, args.height),
    )
    h5_res.close()
    frm2video('.\\frames\\'+name, summary, vid_writer,summaryProgress,GenerateVideo)
    vid_writer.release()
    #Generate play on front end
    #if(videolabel!=None):
    #    player = tkvideo(args.save_dir+'\\'+name+'.mp4', videolabel, loop=1, size=(240, 240))
    #    player.play()


if __name__ == '__main__':
    generateVideoSummary()