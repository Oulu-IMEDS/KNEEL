# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import glob
import deepdish as dd
from torch.utils import data

import torchvision.transforms as standard_transforms
from ..proj_utils.local_utils import overlay_bbox

class Knee(data.Dataset):
    def __init__(self, data_root, mode, transform=None):
        self.data_root = data_root
        assert mode in ["train", "val", "test", "most"], "Unknown mode: {}".format(mode)
        self.mode = mode
        self.data_path = os.path.join(data_root, "H5", mode + "H5")
        self.item_list = glob.glob(os.path.join(self.data_path, "*.h5"))
        self.item_num  = len(self.item_list)
        self.transform = transform
        self._classes = ["0", "1", "2", "3", "4"]

    def __getitem__(self, index):
        # get bone info
        cur_item = dd.io.load(self.item_list[index])
        # get image
        cur_img = cur_item["images"].astype(np.float32)
        if self.transform is not None:
            cur_img = self.transform(cur_img)


        cur_boxes = np.asarray(cur_item["gt_boxes"])
        cur_classes = np.asarray(cur_item["gt_classes"])
        cur_name = cur_item["origin_im"]

        return cur_img, cur_boxes, cur_classes, cur_name

    def __len__(self):
        return self.item_num

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def item_names(self):
        return [os.path.basename(ele) for ele in self.item_list]

    @property
    def num_items(self):
        return self.item_num

    def get_all_bbox(self):
        bbox_all = []
        for idx in range(self.item_num):
            cur_item = dd.io.load(self.item_list[idx])
            bbox = cur_item["gt_boxes"]
            bbox_all.extend(bbox)

        return np.asarray(bbox_all)

    def get_mean_pixel(self):
        gray_list = np.zeros((self.item_num, ), np.float32)
        for idx in range(self.item_num):
            cur_item = dd.io.load(self.item_list[idx])
            cur_img = cur_item["images"]
            gray_list[idx] = np.mean(cur_img) / 255.0

        pixel_mean = np.mean(gray_list)
        return pixel_mean

    def get_var_pixel(self):
        pixel_arr = np.zeros((self.item_num, 256, 320), np.float32)

        for idx in range(self.item_num):
            cur_item = dd.io.load(self.item_list[idx])
            cur_img = cur_item["images"]

            var_img = cur_img[:, :, 0] / 255.0
            pixel_arr[idx] = var_img
        pixel_var = np.var(pixel_arr)
        return pixel_var

    def overlayImgs(self, save_path):
        for idx in range(self.item_num):
            cur_item = dd.io.load(self.item_list[idx])
            cur_img = cur_item["images"]
            cur_boxes = cur_item["gt_boxes"]
            cur_classes = cur_item["gt_classes"]
            cur_name = cur_item["origin_im"]

            overlay_img = overlay_bbox(cur_img, cur_boxes, len=3).astype(np.uint8)

            for ind, box in enumerate(cur_boxes):
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 255, 0)
                col_mid = int((box[2] + box[0]) / 2.0)
                row_mid = int((box[3] + box[1]) / 2.0)
                cv2.putText(overlay_img, str(cur_classes[ind]), (col_mid, row_mid), font, 1, color, 2)

            cv2.imwrite(os.path.join(save_path, cur_name+".png"), overlay_img)


if __name__ == "__main__":
    data_root = "../../data/"
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize([0.5]*3, [0.5]*3)])
    val_dataset = Knee(data_root, "validation", transform=input_transform)
    val_dataloader = data.DataLoader(val_dataset, batch_size=4)

    epoch_num = 10
    for idx_epoch in range(epoch_num):
        for cur_batch, data in enumerate(val_dataloader):
            cur_img, cur_boxes, cur_classes, cur_name = data
            import pdb; pdb.set_trace()
            print("In Epoch: {}, Batch: {}".format(idx_epoch, cur_batch))
