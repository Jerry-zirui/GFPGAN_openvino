import torch

from facexlib.utils import load_file_from_url
#from .bisenet import BiSeNet
from facexlib.parsing.parsenet import ParseNet
from .torch_ov_model import torch_model

def init_parsing_model(model_name='bisenet', half=False, model_rootpath=None):
    device='cuda'

    if model_name == 'bisenet':
        raise NotImplementedError("bisenet not tested")
        model = BiSeNet(num_class=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()

    example = (torch.randn(1, 3, 512, 512),)
    ov_model = torch_model(model, example)

    if model:
        del model

    return ov_model
