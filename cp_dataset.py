#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
import os
import cv2
from scipy import ndimage
from random import randrange

train_cloth_ids = np.load('train_cloth_ids.npy').tolist()
test_cloth_ids = np.load('test_cloth_ids.npy').tolist()


def cloth_region_position(new_image):
    if len(new_image.shape) == 3:
        fill_pix_bool = np.all(new_image == (1,1,1), axis=-1)
    else:
        fill_pix_bool = new_image == 1

    fill_pix_indices = np.where(fill_pix_bool)
    fill_pix_indices = np.squeeze(np.array([fill_pix_indices])).T
    return fill_pix_indices


def inner_distance(segmentation_one_channel):
    inner_dis = ndimage.distance_transform_edt(segmentation_one_channel)
    return inner_dis


def outer_distance(segmentation_one_channel):
    outer_dis = ndimage.distance_transform_edt(np.logical_not(segmentation_one_channel))
    return outer_dis

def shrink(image, cloth_mask, pixel=-2):
    print('image_shape_pre',image.shape)
    new_image = image.copy()
    new_image = np.swapaxes(new_image, 0, 1)
    new_image = np.swapaxes(new_image, 1, 2)
    print('image_shape_post',image.shape)

    fill_pix_indices = cloth_region_position(new_image)
    print (image.shape, cloth_mask.shape)
    # new_image[fill_pix_indices[:,0],fill_pix_indices[:,1]] = np.array([0.,0.,0.])
    
    # segmentation = np.zeros_like(new_image)[:,:,0]
    # segmentation[fill_pix_indices[:,0], fill_pix_indices[:,1]] = 1
    inner_dis = inner_distance(cloth_mask)
    outer_dis = outer_distance(cloth_mask)
    combine_dis = inner_dis*(-1)+outer_dis
    
    shrink_idx = combine_dis<=pixel
    shrink_idx = np.where(shrink_idx)
    shrink_idx = np.squeeze(np.array([shrink_idx])).T
    
    new_image[shrink_idx[:,0],shrink_idx[:,1],:] = np.array([128,128,128])
    
    return new_image

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        if self.datamode == 'test':
            test_cloth_ids.remove(c_name)
            lenth = len(test_cloth_ids)
            unpair_c_name = list(test_cloth_ids)[randrange(lenth)]
        else:
            train_cloth_ids.remove(c_name)
            lenth = len(train_cloth_ids)
            unpair_c_name = list(train_cloth_ids)[randrange(lenth)]

        # cloth image & cloth mask
        # if self.stage == 'GMM':
        #     c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        #     # cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        # else:
        #     c = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
        #     # cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
        c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        c = self.transform(c)  # [-1,1]

        c_unpair = Image.open(osp.join(self.data_path, 'cloth', unpair_c_name))
        c_unpair = self.transform(c_unpair)  # [-1,1]

        # cm_array = np.array(cm)
        # cm_array = (cm_array >= 128).astype(np.float32)
        # cm = torch.from_numpy(cm_array) # [0,1]
        # cm.unsqueeze_(0)

        # person image 
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(im_parse)
        if len(np.array(im_parse).shape)==3:
            parse_array = parse_array[:,:,0]
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
        parse_cloth_top = (parse_array == 5).astype(np.float32)


        parse_upper = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32) + \
                (parse_array == 14).astype(np.float32) + \
                (parse_array == 15).astype(np.float32)

        ##### prepare the dilated image
        im_array = np.array(im)
        dilated_upper_wuton = shrink(im_array, parse_upper, pixel=9)
        dilated_upper_wuton = Image.fromarray(dilated_upper_wuton)
        dilated_upper_wuton = self.transform(dilated_upper_wuton) # [-1,1]

        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        print('parse_shape_post', parse_shape.shape)
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]
        pcm_top = Image.fromarray((parse_cloth_top*255).astype(np.uint8))
        pcm_top = self.transform(pcm_top) # [-1,1]
         
        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform(im_pose)
        
        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0) 
        agnostic_cloth = torch.cat([shape, im_h, pose_map, pcm_top], 0) 

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            # 'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            # 'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            # 'shape': shape,         # for visualization
            # 'head': im_h,           # for visualization
            # 'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            # 'top_cloth_parse': pcm_top,
            # 'agnostic_cloth': agnostic_cloth,
            'dilated_upper_wuton': dilated_upper_wuton,
            'c_unpaired': c_unpair,
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

