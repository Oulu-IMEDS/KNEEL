# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import torch
import datetime
import cv2

from .proj_utils.plot_utils import plot_scalar
from .proj_utils.torch_utils import to_device, set_lr
from .proj_utils.local_utils import mkdirs


def train_eng(train_dataloader, val_dataloader, model_root, net, args):
    net.train()
    lr = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    loss_train_plot = plot_scalar(name = "loss_train", env= args.model_name, rate = args.display_freq)
    loss_bbox_plot = plot_scalar(name = "loss_bbox",   env= args.model_name, rate = args.display_freq)
    loss_iou_plot = plot_scalar(name = "loss_iou",     env= args.model_name, rate = args.display_freq)
    loss_cls_plot = plot_scalar(name = "loss_cls",     env= args.model_name, rate = args.display_freq)

    print("Start training...")
    best_loss = 1.0e6 # given a high loss value
    for cur_epoch in range(1, args.maxepoch+1):
        train_loss, bbox_loss, iou_loss, cls_loss = 0., 0., 0., 0.
        mini_batch_num = 0
        for cur_batch, data in enumerate(train_dataloader):
            cur_imgs, cur_boxes, cur_classes, cur_names = data
            # forward
            im_data = to_device(cur_imgs, net.device_id)
            _ = net(im_data, cur_boxes, cur_classes)

            # backward
            loss = net.loss
            bbox_loss_val  = net.bbox_loss.data.cpu().numpy().mean()
            iou_loss_val   = net.iou_loss.data.cpu().numpy().mean()
            cls_loss_val   = net.cls_loss.data.cpu().numpy().mean()
            train_loss_val = loss.data.cpu().numpy().mean()

            bbox_loss  += bbox_loss_val
            iou_loss   += iou_loss_val
            cls_loss   += cls_loss_val
            train_loss += train_loss_val
            loss_train_plot.plot(train_loss_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mini_batch_num += 1
            # Print error information
            if (cur_batch + 1) % args.display_freq == 0:
                num_samples = args.display_freq * args.batch_size
                ttl_loss = train_loss/num_samples
                bb_loss, iou_loss = bbox_loss/num_samples, iou_loss/num_samples
                cls_loss = cls_loss/num_samples
                print_str = "Train:Epoch:{:>3}/{:>3}, {:>3}/{:>3} total loss:{:.6f}, bbox_loss:{:.6f}, iou_loss:{:.6f}, cls_loss:{:.2f}"
                print(print_str.format(cur_epoch, args.maxepoch, mini_batch_num, len(train_dataloader), ttl_loss, bb_loss, iou_loss, cls_loss))
                train_loss, bbox_loss, iou_loss, cls_loss = 0, 0., 0., 0.

        if cur_epoch % args.save_freq == 0:
            # Validate current model's performance
            best_loss = validate(val_dataloader, net, cur_epoch, model_root, best_loss, args)
            net.train()

        # Adjust learing rate
        if cur_epoch in args.lr_decay_epochs:
            lr *= args.lr_decay
            optimizer = set_lr(optimizer, lr)
            print("Current lr is {}".format(lr))


def validate(val_dataloader, net, cur_epoch, model_root, best_loss, args):
    net.eval()
    bbox_loss_val, iou_loss_val, cls_loss_val, total_loss_val = 0., 0., 0., 0.

    for ind, data in enumerate(val_dataloader):
        cur_imgs, cur_boxes, cur_classes, cur_names = data
        # forward
        im_data = to_device(cur_imgs, net.device_id)
        _ = net(im_data, cur_boxes, cur_classes)
        bbox_loss_val  += net.bbox_loss.data.cpu().numpy().mean()
        iou_loss_val   += net.iou_loss.data.cpu().numpy().mean()
        cls_loss_val   += net.cls_loss.data.cpu().numpy().mean()
        total_loss_val += net.loss.data.cpu().numpy().mean()

    total_loss_val /= (ind + 1)

    # Saving new model
    if total_loss_val < best_loss:
        best_loss = total_loss_val

        bbox_loss_val /= (ind + 1)
        iou_loss_val /= (ind + 1)
        cls_loss_val /= (ind + 1)

        print("Validation--Epoch {:>2}, total loss: {:.5f}, bbox_loss: {:.5f}, iou_loss: {:.5f}, cls_loss:{:.2f}".format(
            cur_epoch, total_loss_val, bbox_loss_val, iou_loss_val, cls_loss_val))

        weights_name = "det-epoch-" + str(cur_epoch).zfill(3) + "-" + "{:.6f}".format(best_loss) + '.pth'
        model_path = os.path.join(model_root, weights_name)
        torch.save(net.state_dict(), model_path)
        print('save weights at {}'.format(model_path))

    return best_loss
