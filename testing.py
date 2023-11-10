import torch as th
import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader

import torch.utils.data as data
import utils.utils_video as utils_video
from rvrt.original import net
from utils import utils_image as util
from main_test_rvrt import test_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='001_RVRT_videosr_bi_REDS_30frames', help='tasks: 001 to 006')
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--folder_lq', type=str, default='testsets/REDS4/sharp_bicubic',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs='+', default=[0,256,256],
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2,20,20],
                        help='Overlapping of different tiles')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers in data loading')
    parser.add_argument('--save_result', action='store_true', help='save resulting image')
    args = parser.parse_args()

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- benchmarking --
    tasks =  ['001_RVRT_videosr_bi_REDS_30frames',
              '002_RVRT_videosr_bi_Vimeo_14frames',
              '004_RVRT_videodeblurring_DVD_16frames',
              '005_RVRT_videodeblurring_GoPro_16frames',
              '006_RVRT_videodenoising_DAVIS_16frames']
    nchannels = [3,3,3,3,4,4]
    results = []
    for task,chnls in zip(tasks,nchannels):
        if not("deno" in task): continue

        # -- shape --
        if "videosr" in task:
            _task = "sr"
        elif "videodeblurring" in task:
            _task = "deblur"
        elif "videodenoising" in task:
            _task = "deno"
            # print(model)
        else:
            _task = "idk"
        if _task == "sr":
            vshape = (1,5,chnls,156,156)
        else:
            # vshape = (4,4,chnls,256,256)
            # vshape = (1,3,chnls,512,512)
            # vshape = (1,85,chnls,540,960)
            # vshape = (1,20,chnls,256,256)
            # vshape = (1,85,chnls,540,960)
            vshape = (2,8,chnls,156,156)
        if _task == "sr": continue

        # -- view --
        args.task = task
        model = get_model(args)
        model.eval()
        model = model.to(device)

    save_dir = f'results/{args.task}'
    if args.save_result:
        os.makedirs(save_dir, exist_ok=True)

    test_set = SingleVideoRecurrentTestDataset({'dataroot_lq':args.folder_lq,
                                              'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)

    for idx, batch in enumerate(test_loader):
        lq = batch['L'].to(device)
        folder = batch['folder']
        gt = batch['H'] if 'H' in batch else None

        # inference
        with torch.no_grad():
            output = test_video(lq, model, args)

        if 'vimeo' in args.folder_lq.lower():
            output = (output[:, 3:4, :, :, :] + output[:, 10:11, :, :, :]) / 2
            batch['lq_path'] = batch['gt_path']

        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        test_results_folder['psnr_y'] = []
        test_results_folder['ssim_y'] = []

        for i in range(output.shape[1]):
            # save image
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
            if args.save_result:
                seq_ = osp.basename(batch['lq_path'][i][0]).split('.')[0]
                os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}.png', img)


def get_model(args):
    ''' prepare model and dataset according to args.task. '''

    # define model
    if args.task == '001_RVRT_videosr_bi_REDS_30frames':
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12,
                    attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['002_RVRT_videosr_bi_Vimeo_14frames', '003_RVRT_videosr_bd_Vimeo_14frames']:
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['004_RVRT_videodeblurring_DVD_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['005_RVRT_videodeblurring_GoPro_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12,
                    attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['GoPro11-part1', 'GoPro11-part2']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False


    elif args.task == '006_RVRT_videodenoising_DAVIS_16frames':
        model = net(upscale=1, clip_size=2, img_size=[-1, 2, 11],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12,
                    attention_heads=12, attention_window=[3, 3],
                    nonblind_denoising=True, cpu_cache_length=100)
        datasets = ['Set8', 'DAVIS-test']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = True
    return model



class SingleVideoRecurrentTestDataset(data.Dataset):
    """Single video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames (only input LQ path).

    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(SingleVideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.lq_root = opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq = {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))

        for subfolder_lq in subfolders_lq:
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))

            max_idx = len(img_paths_lq)

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
        else:
            imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])

        return {
            'L': imgs_lq,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)


if __name__ == '__main__':
    main()
