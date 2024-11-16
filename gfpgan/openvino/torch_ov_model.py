import openvino as ov
import torch

def flattenize_structure(outputs):
    if not isinstance(outputs, dict):
        outputs = flattenize_tuples(outputs)
        return [i.numpy(force=True) if isinstance(i, torch.Tensor) else i for i in outputs]
    else:
        return dict((k, v.numpy(force=True) if isinstance(v, torch.Tensor) else v) for k, v in outputs.items())

def flattenize_tuples(list_input):
    if not isinstance(list_input, (tuple, list)):
        return [list_input]
    unpacked_pt_res = []
    for r in list_input:
        unpacked_pt_res.extend(flattenize_tuples(r))
    return unpacked_pt_res

class torch_model():
    def __init__(self, model:torch.nn.Module, example) -> None:
        self.example = example
        self.ov_model = self.convert_model(model, example)
        pass

    def prepare_inputs(self, example):
        inputs = example
        if isinstance(inputs, dict):
            return dict((k, v.numpy()) for k, v in inputs.items())
        else:
            return flattenize_structure(inputs)

    def convert_model(self, model_obj, example):
        print("Convert the model into ov::Model")
        shape_list = example[0].shape
        shape_list = list(shape_list)
        print(f"inputs shape: {shape_list}")

        return ov.convert_model(
            model_obj, example_input=example, input=shape_list, verbose=True)

    def infer_ov_model(self, inputs, ie_device = None):
        ov_inputs = self.prepare_inputs(inputs)
        core = ov.runtime.Core()
        compiled = core.compile_model(self.ov_model, ie_device)
        ov_outputs = compiled(ov_inputs)
        return ov_outputs