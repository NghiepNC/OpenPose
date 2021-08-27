#!/usr/bin/env python
# coding: utf-8

# # Thực hiện học trên model
# 

# In[1]:


# import
import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


# Thiết định các giá trị ban đầu
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# # Tạo DataLoader

# In[4]:


from utils.dataloader import make_datapath_list, DataTransform, COCOkeypointsDataset

# Tạo list từ MS COCO
train_img_list, train_mask_list, val_img_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(
    rootpath="./data/")
print(len(train_img_list))
print(len(train_meta_list))

# ★Lấy 1024 data để train for cpu
# lấy 10240 data for gpu
data_num = 10240  # bội số của batch size
train_img_list = train_img_list[:data_num]
train_mask_list = train_mask_list[:data_num]
val_img_list = val_img_list[:data_num]
val_mask_list = val_mask_list[:data_num]
train_meta_list = train_meta_list[:data_num]
val_meta_list = val_meta_list[:data_num]

# Tạo dataset
train_dataset = COCOkeypointsDataset(
    val_img_list, val_mask_list, val_meta_list, phase="train", transform=DataTransform())

# Để đơn giản hóa trong bài này không tạo dữ liệu đánh giá
# val_dataset = CocokeypointsDataset(val_img_list, val_mask_list, val_meta_list, phase="val", transform=DataTransform())

# Tạo DataLoader
batch_size = 8

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
dataloaders_dict = {"train": train_dataloader, "val": None}


# # Tạo Model 

# In[5]:


from utils.openpose_net import OpenPoseNet
net = OpenPoseNet()


# # Định nghĩa hàm mất mát

# In[6]:


class OpenPoseLoss(nn.Module):
    def __init__(self):
        super(OpenPoseLoss, self).__init__()

    def forward(self, saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask):
        """
        tính loss
        Parameters
        ----------
        saved_for_loss : Output ofOpenPoseNet (list)

        heatmap_target : [num_batch, 19, 46, 46]
            Anotation information

        heatmap_mask : [num_batch, 19, 46, 46]
            

        paf_target : [num_batch, 38, 46, 46]
            PAF Anotation

        paf_mask : [num_batch, 38, 46, 46]
            PAF mask

        Returns
        -------
        loss : 
        """

        total_loss = 0
        
        for j in range(6):

            # Không tính những vị trí của mask
            pred1 = saved_for_loss[2 * j] * paf_mask
            gt1 = paf_target.float() * paf_mask

            # heatmaps
            pred2 = saved_for_loss[2 * j + 1] * heat_mask
            gt2 = heatmap_target.float()*heat_mask

            total_loss += F.mse_loss(pred1, gt1, reduction='mean') +                 F.mse_loss(pred2, gt2, reduction='mean')

        return total_loss


criterion = OpenPoseLoss()


# # Thiết định optimizer

# In[7]:


optimizer = optim.SGD(net.parameters(), lr=1e-2,
                      momentum=0.9,
                      weight_decay=0.0001)


# # Thực hiện việc học

# In[8]:


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # Xem máy train của bạn có dùng gpu hay không
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use：", device)

    # chuyển thông tin model vào ram
    net.to(device)

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloaders_dict["train"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    iteration = 1

    # vòng học
    for epoch in range(num_epochs):

        # lưu thời gian bắt đầu học
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  
        epoch_val_loss = 0.0  

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # phân loại data học và kiểm chứng
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  
                optimizer.zero_grad()
                print('（train）')

            # lần này bỏ qua thông tin kiểm chứng
            else:
                continue
                # net.eval()   
                # print('-------------')
                # print('（val）')

            # Lấy từng minibatch files từ data loader
            for imges, heatmap_target, heat_mask, paf_target, paf_mask in dataloaders_dict[phase]:
                if imges.size()[0] == 1:
                    continue

                # Gửi data đến GPU nếu máy cài GPU
                imges = imges.to(device)
                heatmap_target = heatmap_target.to(device)
                heat_mask = heat_mask.to(device)
                paf_target = paf_target.to(device)
                paf_mask = paf_mask.to(device)

                # thiết lập giá trị khởi tạo cho optimizer
                optimizer.zero_grad()

                # tính forward
                with torch.set_grad_enabled(phase == 'train'):
                    _, saved_for_loss = net(imges)

                    loss = criterion(saved_for_loss, heatmap_target,
                                     heat_mask, paf_target, paf_mask)
                    del saved_for_loss
                    # gửi thông tin loss theo back propagation khi học
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 2 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {} || Loss: {:.4f} || 2iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # Validation (skip)
                    # else:
                        #epoch_val_loss += loss.item()

        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, 0))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    # Lưu thông tin sau khi học
    torch.save(net.state_dict(), 'weights/openpose_net_' +
               str(epoch+1) + '.pth')


# In[ ]:


# HỌc (chạy 1 lần)
num_epochs = 2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)


# In[ ]:




