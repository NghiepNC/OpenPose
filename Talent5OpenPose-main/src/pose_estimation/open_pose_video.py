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
        cv2.imwrite('./data/dance_' + str(frame_index) + '.jpg', frame)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return ret_val


# Step2: tạo các file ảnh dự đoán
def predict(img_file_paths: typing.List, model_path: str) -> int:
    """
    step2: predict với những ảnh đã tạo ra
    """
    # create model to predict
    net = OpenPoseNet()

    net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    keys = list(net_weights.keys())

    weights_load = {}

    for key in range(len(keys)):
        weights_load[list(net.state_dict().keys())[key]] \
            = net_weights[list(keys)[key]]

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    for image_path in img_file_paths:
        print("predicting {}".format(image_path))
        result_img = _predict_one_image(net, image_path)

        cv2.imwrite('./data/predicted_images/{}'.
                    format(os.path.basename(image_path)),
                    result_img)


def create_predicted_video(video_path, images_path) -> str:
    predicted_video_path = ""
    first_img = cv2.imread(images_path[0])

    height, width, layers = first_img.shape
    size = (width, height)
    fps = 25

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                            fps, size)

    # để viết image đúng thứ tự cần phải sort là images_path
    images_path = sorted(images_path, key=lambda x: int(x.split("_")[2].split(".")[0]))

    for image_path in images_path:
        print("writing video {}".format(image_path))
        image = cv2.imread(image_path)
        video.write(image)

    video.release()


def _predict_one_image(net: object, image_path: str) -> object:
    """
    predict one image
    """
    ori_img = cv2.imread(image_path)

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
    # extract from video to image
    extract_video('./data/dance.mp4')

    # create predicted image
    img_file_paths = glob.glob("./data/dance_*")
    ret = predict(img_file_paths=img_file_paths,
                  model_path='./weights/pose_model_scratch.pth')

    # create video
    predicted_images_path = glob.glob("./data/predicted_images/dance_*")
    create_predicted_video("./data/predicted_video.avi",
                           predicted_images_path)

