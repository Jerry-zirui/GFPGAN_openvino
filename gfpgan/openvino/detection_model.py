import torch
from copy import deepcopy

from .torch_ov_model import torch_model
from facexlib.utils import load_file_from_url
from facexlib.detection.retinaface import RetinaFace

import openvino as ov

def infer_ov_model(ov_model, inputs, ie_device):
    core = ov.runtime.Core()
    compiled = core.compile_model(ov_model, ie_device)
    ov_outputs = compiled(inputs)
    return ov_outputs

def init_detection_model(model_name, half=False, model_rootpath=None):
    device='cpu'

    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
    elif model_name == 'retinaface_mobile0.25':
        raise NotImplementedError ("Retinaface_mobile0.25 not tested")
        model = RetinaFace(network_name='mobile0.25', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)

    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()

    example = (torch.randn(1, 3, 1156, 789),)
    ov_model = torch_model(model, example)

    if model:
        del model

    return ov_model
