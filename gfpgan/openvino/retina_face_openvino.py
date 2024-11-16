import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models._utils import IntermediateLayerGetter as IntermediateLayerGetter

from facexlib.detection.align_trans import get_reference_facial_points

from facexlib.detection.retinaface_utils import (PriorBox, decode, decode_landm,
                                                 py_cpu_nms)
from gfpgan.openvino.torch_ov_model import torch_model
def generate_config(network_name):

    cfg_mnet = {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'return_layers': {
            'stage1': 1,
            'stage2': 2,
            'stage3': 3
        },
        'in_channel': 32,
        'out_channel': 64
    }

    cfg_re50 = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'return_layers': {
            'layer2': 1,
            'layer3': 2,
            'layer4': 3
        },
        'in_channel': 256,
        'out_channel': 256
    }
    if network_name == 'mobile0.25':
        return cfg_mnet
    elif network_name == 'resnet50':
        return cfg_re50
    else:
        raise NotImplementedError(f'network_name={network_name}')

class RetinaFaceOpenVINO:

    def __init__(self, network_name='resnet50', device='CPU', ov_model:torch_model=None):
        self.device = device
        self.cfg = generate_config(network_name)
        self.phase = 'test'
        self.target_size, self.max_size = 1600, 2150
        self.resize, self.scale, self.scale1 = 1., None, None
        self.mean_tensor = np.array([[[[104.]], [[117.]], [[123.]]]])
        self.reference = get_reference_facial_points(default_square=True)
        self.compiled_model:torch_model = ov_model


    def __detect_faces(self, inputs):
        print(f"__detect_faces {inputs.shape}")
        # get scale
        height, width = inputs.shape[2:]
        self.scale = torch.tensor([width, height, width, height], dtype=torch.float32, device='cpu')
        tmp = [width, height, width, height, width, height, width, height, width, height]
        self.scale1 = torch.tensor(tmp, dtype=torch.float32, device='cpu')

        # forawrd

        inputs = inputs.numpy()
        print(f"infer {inputs.shape}")
        results = self.compiled_model.infer_ov_model(inputs)
        #results = self.compiled_model([inputs])
        loc, conf, landmarks = results[0], results[1], results[2]

        # get priorbox
        priorbox = PriorBox(self.cfg, image_size=inputs.shape[2:])
        priors = priorbox.forward()

        return torch.from_numpy(loc), torch.from_numpy(conf), torch.from_numpy(landmarks), priors

    # single image detection
    def transform(self, image, use_origin_size):
        # convert to opencv format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.astype(np.float32)

        # testing scale
        im_size_min = np.min(image.shape[0:2])
        im_size_max = np.max(image.shape[0:2])
        resize = float(self.target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        resize = 1 if use_origin_size else resize

        # resize
        if resize != 1:
            image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        # convert to torch.tensor format
        # image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)

        return image, resize

    def detect_faces(
        self,
        image,
        conf_threshold=0.8,
        nms_threshold=0.4,
        use_origin_size=True,
    ):
        print(f"self.transform {image.shape}, {use_origin_size}")
        image, self.resize = self.transform(image, use_origin_size)
        print(f"self.transform {image.shape}, {self.resize}")

        image = image - self.mean_tensor

        loc, conf, landmarks, priors = self.__detect_faces(image)
        print(f" {type(loc)}, {type(conf)}, {type(landmarks)}, {type(priors)} ")

        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg['variance'])
        boxes = boxes * self.scale / self.resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landmarks = decode_landm(landmarks.squeeze(0), priors, self.cfg['variance'])
        landmarks = landmarks * self.scale1 / self.resize
        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # do NMS
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(bounding_boxes, nms_threshold)
        bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
        # self.t['forward_pass'].toc()
        # print(self.t['forward_pass'].average_time)
        # import sys
        # sys.stdout.flush()
        return np.concatenate((bounding_boxes, landmarks), axis=1)