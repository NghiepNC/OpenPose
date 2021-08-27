# Input: data/dance.mp4
# Output: data/pose_dance.mp4

# standard libs
import typing
import os
from pathlib import Path

# third party libs
import cv2
import torch

import numpy as np

# Internal libs
from utils.openpose_net import OpenPoseNet

# define constant
OK = 0
NG = -1


# Step1: tách lấy file ảnh từ videl
def extract_video(file_path: str) -> int:
    """
    tách lấy file ảnh từ videl
    """
    print("start extract_video")

    # initialize return value
    ret_val = OK

    # check arguments
    assert os.path.exists(file_path)

    cap = cv2.VideoCapture(file_path)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('ext/dance_' + str(frame_index) + '.jpg', frame)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return ret_val


def predict(img_file_paths: str, model_path: str) -> int:
    """
    step2: predict với những ảnh đã tạo ra
    """
    print("start predict")

    # create model to predict
    net = OpenPoseNet()

    net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    keys = list(net_weights.keys())

    weights_load = {}

    for key in range(len(keys)):
        print(key)
        weights_load[list(net.state_dict().keys())[key]] \
            = net_weights[list(keys)[key]]

    print(weights_load)

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)


    print("xu ly anh")

    for path in Path(img_file_paths).glob('dance*.jpg'):

        oriImg = cv2.imread(img_file_paths+"/"+path.name)  # B,G,R
        print(path.name, "da doc")

        # BGR->RGB
        #oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        #cv2.imshow("cap",oriImg)
        #cv2.waitKey(1)
        # plt.show()

        # Resize
        size = (368, 368)
        img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.

        # chuẩn hóa
        color_mean = [0.485, 0.456, 0.406]
        color_std = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

        # （height 、width、colors）→（colors、height、width）
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

        # cho thông tin vào tensor
        img = torch.from_numpy(img)

        x = img.unsqueeze(0)

        print(path.name,"da xu ly")

        # Tạo heatmap
        net.eval()
        predicted_outputs, _ = net(x)

        pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
        heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

        pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

        pafs = cv2.resize(
            pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(
            heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        print(path.name, "da tao heat")

        print("bat dau du doan")

        from utils.decode_pose import decode_pose
        _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)

        cv2.imwrite("exp/"+path.name,result_img)





# Step3: nối những ảnh đã predict xong thành 1 video
def compose_predicted_images(file_paths):
    img_array = []
    img_name = []

    for path in Path(file_paths).glob('dance*.jpg'):
        img_name.append(path.name)

    img_name.sort(key=len)

    for i in img_name:
        img = cv2.imread(file_paths+"/"+i)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    print(".", end= "")
    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    #return predicted_video_path


if __name__ == "__main__":
    print("start")
    #ret = extract_video('./data/dance.mp4')
    ret = OK
    print(ret)
    if ret == OK:
        print("hoge")
        ret = predict("ext",
            model_path='./weights/pose_model_scratch.pth')
    compose_predicted_images("exp")
