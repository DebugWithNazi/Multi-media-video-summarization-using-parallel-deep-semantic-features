"""
    Generate Dataset

    1. Converting video to frames
    2. Extracting features
    3. Getting change points
    4. User Summary ( for evaluation )

"""
import os
import math
import gc


import cv2
import numpy as np
import h5py
import scipy.io
import pandas as pd
import torch
import os.path as osp

from tkinter import *
from .KTS.cpd_auto import cpd_auto
from .models.CNN3D import I3D, ResNet3D
from .models.CNN import ResNet, GoogleNet, Inception
from .utils.parse_arguments import parse_arguments_generate_dataset


def array_to_id(number_array):
    number_array = number_array.squeeze()
    chr_array = [chr(x) for x in number_array]
    string_array = ''.join(chr_array)
    return string_array


def get_field_by_idx(all_mat, field, idx):
    key = all_mat['tvsum50'][field][idx][0]
    return np.array(all_mat[key])


def get_video_ids(all_mat):
    video_ids = []
    for video_idx in range(len(all_mat['tvsum50']['video'])):
        video_id_na = get_field_by_idx(all_mat, 'video', video_idx)
        video_id = array_to_id(video_id_na)
        video_ids.append(video_id)
    return video_ids


class Generate_Dataset:
    def __init__(self, video_path, path_ground_truth, save_path, dataset='custom',
                 progressBar=None,
                 progressLabel=None,
                 currentFrame=None,
                 path_weights_flow="..\\datasets\\pytorch-i3d\\models\\flow_imagenet.pt",
                 path_weights_rgb="..\\datasets\\pytorch-i3d\\models\\rgb_imagenet.pt",
                 paht_weights_r3d101_KM="..\\datasets\\3D-ResNets-PyTorch\\weights\\r3d101_KM_200ep.pth",
                 frame_root_path='..\\frames', save_root_path='..\\datasets\\object_features\\', resnet=True,
                 inception=True, googlenet=True):
        self.progressBar = progressBar
        self.currentFrame = currentFrame
        self.batch_Size = 3500
            # 5010
        self.device = torch.device(
            "cuda:" + (osp.getenv('N_CUDA') if os.getenv('N_CUDA') else "0") if torch.cuda.is_available() else "cpu")
        # print(f'Using device {self.device}')
        #  if torch.cuda.is_available():
        # print(f'Using {torch.cuda.get_device_name(0)}')
        self.progressBar['value'] = 40
        self.currentFrame.update()
        self.image_models = self._get_model_frame_feature(resnet=resnet, inception=inception, googlenet=googlenet)
        self.video_models = self._get_model_video(path_weights_flow=path_weights_flow,
                                                  path_weights_rgb=path_weights_rgb,
                                                  paht_weights_r3d101_KM=paht_weights_r3d101_KM)
        self.progressBar['value'] = 60
        self.currentFrame.update()
        self.progressLabel = progressLabel
        self.dataset = dataset
        self.video_list = []
        self.video_path = ''
        self.frame_root_path = frame_root_path
        self.save_root_path = save_root_path
        self.save_path = save_path
        self.h5_file = h5py.File(save_root_path + save_path, 'w')
        self.gt_list = []
        self.gt_path = ''
        self.progressBar['value'] = 80
        self.currentFrame.update()

        # self._set_gt_lista(path_ground_truth, self.dataset)
        self._set_video_list(video_path, self.dataset)
        self.progressBar['value'] = 100
        self.currentFrame.update()

    def _get_model_frame_feature(self, resnet=True, inception=True, googlenet=True):
        image_models = {}
        if resnet:
            resnet = ResNet(self.device)
            image_models["resnet"] = resnet.eval()
        if inception:
            inception = Inception(self.device)
            image_models["inception"] = inception.eval()
        if googlenet:
            googlenet = GoogleNet(self.device)
            image_models["googlenet"] = googlenet.eval()
        return image_models

    def _get_model_video(self, path_weights_flow, path_weights_rgb, paht_weights_r3d101_KM):
        i3d = I3D(self.device, path_weights_flow, path_weights_rgb)
        i3d = i3d.eval()
        resnet3D = ResNet3D(device=self.device, path_weights=paht_weights_r3d101_KM)
        resnet3D = resnet3D.eval()
        video_models = {
            "i3d": i3d,
            "resnet3D": resnet3D,
        }
        return video_models

    def _extract_video_feature(self, frame_resized, flow_frames, currentFrame, progressLabel, progressBar):
        features_rgb, features_flow = self.video_models["i3d"](frame_resized, flow_frames, currentFrame, progressLabel,
                                                               progressBar)
        features_3D = self.video_models["resnet3D"](frame_resized, currentFrame, progressLabel, progressBar)
        return features_rgb, features_flow, features_3D

    def _set_video_list(self, video_path, dataset='summe'):
        if os.path.isdir(video_path):
            self.video_path = video_path
            if dataset in ('summe', 'cosum'):
                self.video_list = [videoname for videoname in os.listdir(video_path) if
                                   videoname.lower().endswith(".mp4")]
                self.video_list.sort()
            elif dataset == 'tvsum':
                self.video_list = [videoname + ".mp4" for videoname in self.gt_list]
            elif dataset in ('ovp', 'youtube'):
                self.video_list = [videoname for videoname in os.listdir(video_path) if
                                   videoname.lower().endswith((".mpg", ".avi", ".flv"))]
                self.video_list = sorted(self.video_list, key=lambda x: int(x.split('.')[0][1:]))
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            if (dataset == "youtube") and (idx in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 21)):
                continue
            self.h5_file.create_group('video_{}'.format(idx + 1))

    def _set_gt_lista(self, path_ground_truth, dataset='summe'):
        if os.path.isdir(path_ground_truth):
            if dataset == 'summe':
                self.gt_path = path_ground_truth
                self.gt_list = [gtvideo for gtvideo in os.listdir(path_ground_truth) if
                                gtvideo.lower().endswith(".mat")]
                self.gt_list.sort()
            elif dataset in ('ovp', 'youtube'):
                self.gt_path = path_ground_truth
                self.gt_list = [gtvideo for gtvideo in os.listdir(path_ground_truth) if
                                os.path.isdir(os.path.join(path_ground_truth, gtvideo))]
                self.gt_list.sort()
            elif dataset == "cosum":
                self.gt_path = path_ground_truth
                df = pd.read_excel(self.gt_path + "\\dataset.xlsx", engine='openpyxl')
                df = df.loc[df["DOWNLOADED"] == 1].reset_index(drop=True)
                self.gt_list = sorted(list(df[["VIDEO_CATEGORY", "SHORT CATEGORY",
                                               "VIDEO_ID_IN_CATEGORY", "VIDEO"]].drop_duplicates().values),
                                      key=lambda x: x[-1] + ".mp4", reverse=False)
        else:
            if dataset == 'summe':
                self.gt_path = ''
                self.gt_list.append(path_ground_truth)
            elif dataset == 'tvsum':
                self.gt_path = h5py.File(path_ground_truth, 'r')
                self.gt_list = get_video_ids(self.gt_path)

    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_feat = {}
        for model in self.image_models.keys():
            frame_feat[model] = self.image_models[model](frame)
        return frame_feat

    def _get_change_points(self, video_feat, n_frame, fps):
        video_feat = video_feat.astype(np.float32)
        seq_len = len(video_feat)
        n_frames = n_frame
        m = int(np.ceil(seq_len / 10 - 1))
        kernel = np.matmul(video_feat, video_feat.T)
        change_points, _ = cpd_auto(kernel, m, 1, verbose=False)
        change_points *= 15
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg

    # TODO : save dataset
    def _save_dataset(self):
        pass

    def _get_ground_truth(self, dataset, gt_path, video_basename, video_feat_for_train, n_frames, fps, gt_info):
        if dataset == 'summe':
            if os.path.isdir(self.gt_path):
                gt_path = os.path.join(self.gt_path, video_basename + ".mat")
            gt_video = scipy.io.loadmat(gt_path)
            gt_video["user_score"] = np.where(gt_video["user_score"] > 0, 1, 0)
        elif dataset == 'tvsum':
            gt_idx = self.gt_list.index(video_basename)
            annotations = get_field_by_idx(self.gt_path, 'user_anno', gt_idx).T
            annotations = (annotations - 1) / 4  # en tvsum el score va de 1 a 4
            gt_video = {'user_score': annotations}
        elif dataset in ('ovp', 'youtube'):
            # youtube gt has less frames than the real ones
            # we found that was resampled to 1.03 fps
            factor = 1 if dataset == 'ovp' else fps / 1.03
            gt_idx = []
            for user_summ in np.sort([folder for folder in os.listdir(os.path.join(self.gt_path, video_basename)) if
                                      os.path.isdir(os.path.join(self.gt_path, video_basename, folder))]):
                list_summ = [int(np.ceil(int(frame.split('.')[0][5:]) * factor)) for frame in
                             os.listdir(os.path.join(self.gt_path, video_basename, user_summ)) if
                             frame.lower().endswith(("png", "jpeg", "jpg"))]
                list_summ.sort()
                gt_idx.append(list_summ)
            m = int(np.ceil(n_frames / (4.5 * fps)))
            kernel = np.matmul(video_feat_for_train.astype(np.float32), video_feat_for_train.astype(np.float32).T)
            change_points, _ = cpd_auto(kernel, m, 1, verbose=False)
            change_points *= 15
            change_points = np.hstack((0, change_points, n_frames))
            begin_frames = change_points[:-1]
            end_frames = change_points[1:]
            change_points = np.vstack((begin_frames, end_frames - 1)).T

            annotations = []
            for user in gt_idx:
                new_gt_user = np.zeros(n_frames)
                segments = [segment for segment in change_points for frame in user if
                            (frame >= segment[0]) and (frame <= segment[1])]
                for segment in segments:
                    new_gt_user[int(segment[0]):int(segment[1] + 1)] = 1
                annotations.append(new_gt_user)
            annotations = np.array(annotations).T
            gt_video = {'user_score': annotations}

        elif dataset == "cosum":
            category, short_name, video_id, video_name = gt_info
            if video_basename == video_name:
                path_ant_category = os.path.join(self.gt_path, "annotation", category)
                path_shots_category = os.path.join(self.gt_path, "shots", category)

                shots = open(os.path.join(path_shots_category, f'{short_name}{video_id}_shots.txt'), 'r')
                shots = shots.read().splitlines()
                shots = [int(np.ceil(int(nframe) * n_frames / int(shots[-1]))) for nframe in shots]

                user1 = scipy.io.loadmat(os.path.join(path_ant_category, f'{video_id}__dualplus.mat'))["labels"][:, 0]
                user2 = scipy.io.loadmat(os.path.join(path_ant_category, f'{video_id}__kk.mat'))["labels"][:, 0]
                user3 = scipy.io.loadmat(os.path.join(path_ant_category, f'{video_id}__vv.mat'))["labels"][:, 0]

                annotations = []
                for user in (user1, user2, user3):
                    new_gt_user = np.zeros(n_frames)
                    for indexshot in user:
                        if indexshot >= len(shots):
                            indexshot = len(shots) - 1
                    new_gt_user[int(shots[int(indexshot) - 1]) - 1:int(shots[int(indexshot)]) - 1] = 1
                    annotations.append(new_gt_user)

                annotations = np.array(annotations).T
                gt_video = {'user_score': annotations}

            else:
                # print("you are no getting the same files")
                sys.exit(0)

        return gt_video

    # @profile
    def generate_dataset(self):
        if (self.progressLabel != None):
            self.progressLabel.config(text='Generating Frames for video')
            self.currentFrame.update()

        for video_idx in range(len(self.video_list)):
            video_filename = self.video_list[video_idx]
            gt_filename = video_filename
            video_path = video_filename

            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_basename = ".".join(os.path.basename(video_path).split('.')[:-1])

            if not os.path.exists(os.path.join(self.frame_root_path, video_basename)):
                os.mkdir(os.path.join(self.frame_root_path, video_basename))

            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            splits = math.ceil(total_frames / self.batch_Size)
            length=splits
            if (splits > 1):
                for current in range(splits):
                    # if(current<=12):
                    #     continue
                    self.progressBar["value"] = 0
                    self.progressLabel.config(text='Generating frames for video')
                    self.currentFrame.update()

                    start_frame = 0
                    end_frame = (current + 1) * self.batch_Size
                    skipNext = False
                    if current+1 == splits and total_frames % self.batch_Size < 500:
                        skipNext = True
                        end_frame += total_frames % self.batch_Size
                    if current > 0:
                        #video_capture = cv2.VideoCapture(video_path)
                        start_frame = current * self.batch_Size
                    try:
                        video_capture = self.generate_frames(video_capture, video_idx, video_basename, start_frame, end_frame, fps, True,
                                         current)
                    except:
                        pass
                    if skipNext:
                        break
                self.marge_splits(length)
            else:
                self.generate_frames(video_capture, video_idx, video_basename, 0, total_frames, fps, False, '')

    def marge_splits(self,splits):
        video_idx = 0
        filename=self.save_root_path + str(0)+'_' + self.save_path
        temp_file = h5py.File(filename, 'r+')

        video_name = temp_file['video_{}'.format(video_idx + 1)]['video_name'][()]
        n_steps = temp_file['video_{}'.format(video_idx + 1)]['n_steps'][()]
        features = temp_file['video_{}'.format(video_idx + 1)]['features'][()]
        features_rn = temp_file['video_{}'.format(video_idx + 1)]['features_rn'][()]
        features_iv3 = temp_file['video_{}'.format(video_idx + 1)]['features_iv3'][()]
        picks = temp_file['video_{}'.format(video_idx + 1)]['picks'][()]
        n_frames = temp_file['video_{}'.format(video_idx + 1)]['n_frames'][()]
        fps = temp_file['video_{}'.format(video_idx + 1)]['fps'][()]
        change_points = temp_file['video_{}'.format(video_idx + 1)]['change_points'][()]
        n_frame_per_seg = temp_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'][()]
        features_rgb = temp_file['video_{}'.format(video_idx + 1)]['features_rgb'][()]
        features_flow = temp_file['video_{}'.format(video_idx + 1)]['features_flow'][()]
        features_3D = temp_file['video_{}'.format(video_idx + 1)]['features_3D'][()]
        temp_file.close()
        # un comment this code once testing is done
        os.remove(filename)
        for index in range(1, splits):
            if not osp.exists(self.save_root_path + str(index) +'_'+ self.save_path):
                continue
            filename=self.save_root_path + str(index) +'_'+ self.save_path
            temp_file = h5py.File(filename, 'r+')
            n_steps = n_steps + temp_file['video_{}'.format(video_idx + 1)]['n_steps'][()]
            features = np.concatenate((features, temp_file['video_{}'.format(video_idx + 1)]['features'][()]))
            features_rn = np.concatenate((features_rn, temp_file['video_{}'.format(video_idx + 1)]['features_rn'][()]))
            features_iv3 = np.concatenate(
                (features_iv3, temp_file['video_{}'.format(video_idx + 1)]['features_iv3'][()]))
            picks = np.concatenate((picks, temp_file['video_{}'.format(video_idx + 1)]['picks'][()]))
            n_frames = n_frames + temp_file['video_{}'.format(video_idx + 1)]['n_frames'][()]
            change_points = np.concatenate(
                (change_points, temp_file['video_{}'.format(video_idx + 1)]['change_points'][()]))
            n_frame_per_seg = np.concatenate(
                (n_frame_per_seg, temp_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'][()]))
            features_rgb = np.concatenate(
                (features_rgb, temp_file['video_{}'.format(video_idx + 1)]['features_rgb'][()]))
            features_flow = np.concatenate(
                (features_flow, temp_file['video_{}'.format(video_idx + 1)]['features_flow'][()]))
            features_3D = np.concatenate((features_3D, temp_file['video_{}'.format(video_idx + 1)]['features_3D'][()]))
            temp_file.close()
            #un comment this code once testing is done
            os.remove(filename)

        change_points, n_frame_per_seg = self._get_change_points(features, n_frames, 15) # reset check point for combined video optionaly you can commit this code
        self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_name
        self.h5_file['video_{}'.format(video_idx + 1)]['n_steps'] = np.array(np.array(list(picks)).shape[0])
        self.h5_file['video_{}'.format(video_idx + 1)]['features'] = list(features)
        self.h5_file['video_{}'.format(video_idx + 1)]['features_rn'] = list(features_rn)
        self.h5_file['video_{}'.format(video_idx + 1)]['features_iv3'] = list(features_iv3)
        self.h5_file['video_{}'.format(video_idx + 1)]['picks'] = list(picks)
        self.h5_file['video_{}'.format(video_idx + 1)]['n_frames'] = n_frames
        self.h5_file['video_{}'.format(video_idx + 1)]['fps'] = fps
        self.h5_file['video_{}'.format(video_idx + 1)]['change_points'] = change_points
        self.h5_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'] = n_frame_per_seg
        self.h5_file['video_{}'.format(video_idx + 1)]['features_rgb'] = features_rgb
        self.h5_file['video_{}'.format(video_idx + 1)]['features_flow'] = features_flow
        self.h5_file['video_{}'.format(video_idx + 1)]['features_3D'] = features_3D

    def generate_frames(self, video_capture, video_idx, video_basename, start_frame, total_frames, fps, is_temp,
                        current_index):
        frame_list = []
        picks = []
        video_feat_for_train = []
        n_frames = 0
        for index in range(start_frame, total_frames):
            success, frame = video_capture.read(index)
           # n_frames=index
            if not success:
                break
            if index % 15 == 0:
                frame_feat = self._extract_feature(frame)
                picks.append(index)
                video_feat_for_train.append(frame_feat)
            img_filename = "{}.jpg".format(str(index + 1))
            print(img_filename, "Current Frame{}".format(str(n_frames + 1)))
            cv2.imwrite(
                os.path.join(self.frame_root_path.strip(), video_basename.strip(), img_filename.strip()).strip(), frame)
            frame_list.append(frame)
            n_frames += 1
            if (self.progressBar != None):
                # print((n_frames/total_frames)*100)
                self.progressBar["value"] = (n_frames / self.batch_Size) * 100
                self.currentFrame.update()

        # progress["value"] = 21
        # print(f'feature images extraction done: in total of {n_frames} frames')
        rate = math.ceil(n_frames / 8500)
        frame_list = frame_list[::rate]
        self.progressLabel.config(text='ReSizing Frames for video')
        self.progressBar["value"] = 0
        self.currentFrame.update()
        totalf = len(frame_list)
        frame_resized = []
        for frame in range(totalf):
            # print('percantage completed : '+str((frame/totalf)*100))
            frame_resized.append(cv2.resize(frame_list[frame], (224, 224)))
            self.progressBar["value"] = (frame / totalf) * 100
            self.currentFrame.update()
        self.progressLabel.config(text='Generating flow frames for video')
        self.progressBar["value"] = 0
        self.currentFrame.update()
        totalf = len(frame_resized)
        flow_frames = []
        for frame in range(totalf):
            # print('percantage completed : '+str((frame/totalf)*100))
            flow_frames.append(cv2.cvtColor(frame_resized[frame], cv2.COLOR_BGR2GRAY))
            self.progressBar["value"] = (frame / totalf) * 100
            self.currentFrame.update()

        # print(f'extracting flow frames ....')

        self.progressLabel.config(text='extracting flow frames ....')
        self.progressBar["value"] = 0
        self.currentFrame.update()
        flow_frames1 = []
        for i in range(len(flow_frames)):
            if i + 16 <= len(flow_frames):
                flow_frames1.append(cv2.calcOpticalFlowFarneback(flow_frames[i], flow_frames[i + 15], None,
                                                                 0.5, 3, 15, 3, 5, 1.2, 0))
            self.progressBar["value"] = (i / len(flow_frames)) * 100
            self.currentFrame.update()
        flow_frames = np.array(flow_frames1)

        rate = 15  # math.ceil(n_frames/8500) #8500 is the limit
        frame_resized = frame_resized[::rate]
        flow_frames = flow_frames[::rate]
        features_rgb, features_flow, features_3D = self._extract_video_feature(frame_resized, flow_frames,
                                                                               self.currentFrame, self.progressLabel,
                                                                               self.progressBar)
        video_feat_for_train_googlenet = np.array([feature["googlenet"] for feature in video_feat_for_train])
        video_feat_for_train_resnet = np.array([feature["resnet"] for feature in video_feat_for_train])
        video_feat_for_train_inception = np.array([feature["inception"] for feature in video_feat_for_train])

        change_points, n_frame_per_seg = self._get_change_points(video_feat_for_train_googlenet, n_frames, fps)
        if is_temp:
            temp_h5_file = h5py.File(self.save_root_path +str(current_index)+'_'+ self.save_path , 'w')
            temp_h5_file.create_group('video_{}'.format(video_idx + 1))
            temp_h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_basename
            temp_h5_file['video_{}'.format(video_idx + 1)]['n_steps'] = np.array(np.array(list(picks)).shape[0])
            temp_h5_file['video_{}'.format(video_idx + 1)]['features'] = list(video_feat_for_train_googlenet)
            temp_h5_file['video_{}'.format(video_idx + 1)]['features_rn'] = list(video_feat_for_train_resnet)
            temp_h5_file['video_{}'.format(video_idx + 1)]['features_iv3'] = list(video_feat_for_train_inception)
            temp_h5_file['video_{}'.format(video_idx + 1)]['picks'] = np.array(list(picks))
            temp_h5_file['video_{}'.format(video_idx + 1)]['n_frames'] = n_frames
            temp_h5_file['video_{}'.format(video_idx + 1)]['fps'] = fps
            temp_h5_file['video_{}'.format(video_idx + 1)]['change_points'] = change_points
            temp_h5_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'] = n_frame_per_seg
            temp_h5_file['video_{}'.format(video_idx + 1)]['features_rgb'] = features_rgb
            temp_h5_file['video_{}'.format(video_idx + 1)]['features_flow'] = features_flow
            temp_h5_file['video_{}'.format(video_idx + 1)]['features_3D'] = features_3D

        else:
            self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_basename
            self.h5_file['video_{}'.format(video_idx + 1)]['n_steps'] = np.array(np.array(list(picks)).shape[0])
            self.h5_file['video_{}'.format(video_idx + 1)]['features'] = list(video_feat_for_train_googlenet)
            self.h5_file['video_{}'.format(video_idx + 1)]['features_rn'] = list(video_feat_for_train_resnet)
            self.h5_file['video_{}'.format(video_idx + 1)]['features_iv3'] = list(video_feat_for_train_inception)
            self.h5_file['video_{}'.format(video_idx + 1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx + 1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx + 1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'] = n_frame_per_seg
            self.h5_file['video_{}'.format(video_idx + 1)]['features_rgb'] = features_rgb
            self.h5_file['video_{}'.format(video_idx + 1)]['features_flow'] = features_flow
            self.h5_file['video_{}'.format(video_idx + 1)]['features_3D'] = features_3D

        del frame_list
        del video_feat_for_train_googlenet
        del video_feat_for_train_resnet
        del video_feat_for_train_inception
        del video_feat_for_train
        del frame_resized
        del flow_frames
        del features_rgb
        del features_flow
        del features_3D
        # del gt_video
        # del user_score
        del change_points
        del n_frame_per_seg
        # del gtscore
        del picks
        gc.collect()
        return video_capture


if __name__ == "__main__":
    args = parse_arguments_generate_dataset()
    videos_path = args.videospath
    groundtruth_path = args.groundtruthpath
    outputname = f'dataset_{args.dataset}_processed.h5'

    if args.dataset not in ('summe', 'tvsum', 'custom', 'ovp', 'youtube', 'cosum'):
        # print("This dataset is not supported for this process")
        sys.exit(0)

    gen = Generate_Dataset(videos_path, groundtruth_path, outputname, args.dataset)

    gen.generate_dataset()
    gen.h5_file.close()
