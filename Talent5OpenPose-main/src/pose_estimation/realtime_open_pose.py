# Input: data/dance.mp4
# Output: data/pose_dance.mp4

# standard libs
import typing
import os

# third party libs
import cv2
import torch
import numpy as np
import glob

# Internal libs
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose


def stream(net):
    """
    stream video from usb camera
    """
    stream = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = stream.read()

        # Output keypoints and the image with the human skeleton blended on it
        output_image = _predict_one_image(net, img)

        # Display the stream
        cv2.putText(output_image, 'OpenPose using Python-OpenCV', (20, 30),
                    font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Human Pose Estimation', output_image)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


def _predict_one_image(net: object, ori_img: object) -> object:
    """
    predict one image
    """
    # Resize
    size = (368, 368)
    img = cv2.resize(ori_img, size, interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.

    # chuẩn hóa
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[
            i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

    # （height 、width、colors）→（colors、height、width）
    img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

    # cho thông tin vào tensor
    img = torch.from_numpy(img)

    x = img.unsqueeze(0)

    # create heat map
    net.eval()
    predicted_outputs, _ = net(x)

    pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
    heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

    pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

    pafs = cv2.resize(
        pafs, (ori_img.shape[1], ori_img.shape[0]),
        interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(
        heatmaps, (ori_img.shape[1], ori_img.shape[0]),
        interpolation=cv2.INTER_CUBIC)

    _, result_img, _, _ = decode_pose(ori_img, heatmaps, pafs)

    return result_img


if __name__ == "__main__":
    # create model to predict
    net = OpenPoseNet()

    net_weights = torch.load('./weights/pose_model_scratch.pth',
                             map_location={'cuda:0': 'cpu'})
    keys = list(net_weights.keys())

    weights_load = {}

    for key in range(len(keys)):
        weights_load[list(net.state_dict().keys())[key]] \
            = net_weights[list(keys)[key]]

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    # stream
    stream(net)




