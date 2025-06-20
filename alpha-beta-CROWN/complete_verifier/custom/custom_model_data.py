#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import hashlib
import time
from typing import Optional, List
import os
import collections
import gzip
from ast import literal_eval
import torch
import numpy as np

import onnx2pytorch
import onnx
import onnxruntime as ort
import onnxoptimizer
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from onnx_opt import compress_onnx

import warnings
import importlib
from functools import partial

import arguments

from model_defs import *
from utils import expand_path


def Customized(def_file, callable_name, *args, **kwargs):
    """Fully customized model or dataloader."""
    if def_file.endswith('.py'):
        # Use relatively path w.r.t. to the configuration file
        if arguments.Config['general']['root_path']:
            path = os.path.join(
                expand_path(arguments.Config['general']['root_path']), def_file)
        elif arguments.Config.file:
            path = os.path.join(os.path.dirname(arguments.Config.file), def_file)
        else:
            path = def_file
        spec = importlib.util.spec_from_file_location('customized', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            # Customized loaders should be in "custom" folder.
            module = importlib.import_module(f'custom.{def_file}')
        except ModuleNotFoundError:
            # If not found, try the current folder.
            module = importlib.import_module(f'{def_file}')
            warnings.warn(  # Old config files may refer to custom loaders in the root folder.
                    f'Customized loaders "{def_file}" should be inside the "custom" folder.')
    # Load model from a specified file.
    model_func = getattr(module, callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def deep_update(d, u):
    """Update a dictionary based another dictionary, recursively.

    (https://stackoverflow.com/a/3233356).
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def unzip_and_optimize_onnx(path, onnx_optimization_flags: Optional[List[str]] = None):
    if onnx_optimization_flags is None:
        onnx_optimization_flags = []
    if len(onnx_optimization_flags) == 0:
        if path.endswith('.gz'):
            onnx_model = onnx.load(gzip.GzipFile(path))
        else:
            onnx_model = onnx.load(path)
        return onnx_model
    else:
        print(f'Onnx optimization with flags: {onnx_optimization_flags}')
        npath = path + '.optimized'
        if os.path.exists(npath):
            print(f'Found existed optimized onnx model at {npath}')
            return onnx.load(npath)
        else:
            print(f'Generate optimized onnx model to {npath}')
            if path.endswith('.gz'):
                onnx_model = onnx.load(gzip.GzipFile(path))
            else:
                onnx_model = onnx.load(path)
            return compress_onnx(onnx_model, path, npath, onnx_optimization_flags, debug=True)


def inference_onnx(path, input):
    # Workaround for onnx bug, see issue #150
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(unzip_and_optimize_onnx(path).SerializeToString(),
                                sess_options=options)
    assert len(sess.get_inputs()) == len(sess.get_outputs()) == 1
    res = sess.run(None, {sess.get_inputs()[0].name: input})[0]
    return res


@torch.no_grad()
def load_model_onnx(path, quirks=None, x=None):
    start_time = time.time()
    onnx_optimization_flags = arguments.Config['model']['onnx_optimization_flags']

    if arguments.Config['model']['cache_onnx_conversion']:
        cached_onnx_suffix = ".cached"
        cached_onnx_filename = f'{path}{cached_onnx_suffix}'

        with open(path, "rb") as file:
            curfile_sha256 = hashlib.sha256(file.read()).hexdigest()

        if os.path.exists(cached_onnx_filename):
            print(f'Loading cached onnx model from {cached_onnx_filename}')
            read_error = False
            try:
                pytorch_model, onnx_shape, old_file_sha256 = torch.load(cached_onnx_filename)
            except (Exception, ValueError, EOFError):
                print("Cannot read cached onnx file. Regenerating...")
                read_error = True
            if not read_error:
                if curfile_sha256 == old_file_sha256:
                    end_time = time.time()
                    print(f'Cached converted model loaded in {end_time - start_time:.4f} seconds')
                    return pytorch_model, onnx_shape
                else:
                    print(f"{cached_onnx_filename} file sha256: {curfile_sha256} does not match the current onnx sha256: {old_file_sha256}. Regenerating...")
        else:
            print(f"{cached_onnx_filename} does not exist.")

    quirks = {} if quirks is None else quirks
    if arguments.Config['model']['onnx_quirks']:
        try:
            config_quirks = literal_eval(arguments.Config['model']['onnx_quirks'])
        except ValueError:
            print('ERROR: onnx_quirks '
                  f'{arguments.Config["model"]["onnx_quirks"]}'
                  'cannot be parsed!')
            raise
        assert isinstance(config_quirks, dict)
        deep_update(quirks, config_quirks)
    print(f'Loading onnx {path} with quirks {quirks}')

    onnx_model = unzip_and_optimize_onnx(path, onnx_optimization_flags)

    if arguments.Config["model"]["input_shape"] is None:
        # find the input shape from onnx_model generally
        # https://github.com/onnx/onnx/issues/2657
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

        if len(net_feed_input) != 1:
            # in some rare case, we use the following way to find input shape
            # but this is not always true (collins-rul-cnn)
            net_feed_input = [onnx_model.graph.input[0]]

        onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
        onnx_shape = tuple(d.dim_value for d in onnx_input_dims)
    else:
        # User specify input_shape
        onnx_shape = arguments.Config['model']['input_shape']

    # remove batch information
    # for nn4sys pensieve parallel, the first dimension of the input size is not batch, do not remove
    if onnx_shape[0] <= 1:
        onnx_shape = onnx_shape[1:]

    try:
        pytorch_model = onnx2pytorch.ConvertModel(
            onnx_model, experimental=True, quirks=quirks)
    except TypeError as e:
        print('\n\nA possible onnx2pytorch version error!')
        print('If you see "unexpected keyword argument \'quirks\'", that indicates your onnx2pytorch version is incompatible.')
        print('Please uninstall onnx2pytorch in your python environment (e.g., run "pip uninstall onnx2pytorch"), and then reinstall using:\n')
        print('pip install git+https://github.com/Verified-Intelligence/onnx2pytorch@fe7281b9b6c8c28f61e72b8f3b0e3181067c7399\n\n')
        print('The error below may not a bug of alpha-beta-CROWN. See instructions above.')
        raise(e)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    conversion_check_result = True
    try:
        # check conversion correctness
        # FIXME dtype of dummy may not match the onnx model, which can cause runtime error
        if x is not None:
            x = x.reshape(1, *onnx_shape)
        else:
            x = torch.randn([1, *onnx_shape])
        output_pytorch = pytorch_model(x).numpy()
        try:
            if arguments.Config['model']['check_optimized']:
                output_onnx = inference_onnx(path+'.optimized', x.numpy())
            else:
                output_onnx = inference_onnx(path, x.numpy())
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument:
            # ONNX model might have shape problems. Remove the batch dimension and try again.
            output_onnx = inference_onnx(path, x.numpy().squeeze(0))
        if 'remove_relu_in_last_layer' in onnx_optimization_flags:
            output_pytorch = output_pytorch.clip(min=0)
        conversion_check_result = np.allclose(
            output_pytorch, output_onnx, 1e-4, 1e-5)
    except:  # pylint: disable=broad-except
        warnings.warn('Not able to check model\'s conversion correctness')
        print('\n*************Error traceback*************')
        import traceback; print(traceback.format_exc())
        print('*****************************************\n')
    if not conversion_check_result:
        print('\n**************************')
        print('Model might not be converted correctly. Please check onnx conversion carefully.')
        print('Output by pytorch:', output_pytorch)
        print('Output by onnx:', output_onnx)
        diff = torch.tensor(output_pytorch - output_onnx).abs().reshape(-1)
        print('Max error:', diff.max())
        index = diff.argmax()
        print('Max error index:', diff.argmax())
        print(f'Output by pytorch at {index}: ',
              torch.tensor(output_pytorch).reshape(-1)[index])
        print(f'Output by onnx at {index}: ',
              torch.tensor(output_onnx).reshape(-1)[index])
        print('**************************\n')

        if arguments.Config["model"]["debug_onnx"]:
            debug_onnx(onnx_model, pytorch_model, x.numpy())

    # TODO merge into the unzip_and_optimize_onnx()
    if arguments.Config["model"]["flatten_final_output"]:
        pytorch_model = nn.Sequential(pytorch_model, nn.Flatten())

    if arguments.Config["model"]["cache_onnx_conversion"]:
        torch.save((pytorch_model, onnx_shape, curfile_sha256), cached_onnx_filename)

    end_time = time.time()
    print(f'Finished onnx model loading in {end_time - start_time:.4f} seconds')

    return pytorch_model, onnx_shape


def debug_onnx(onnx_model, pytorch_model, dummy_input):
    path_tmp = '/tmp/debug.onnx'

    output_onnx = {}
    for node in enumerate_model_node_outputs(onnx_model):
        print('Inferencing onnx node:', node)
        save_onnx_model(select_model_inputs_outputs(onnx_model, node), path_tmp)
        optimized_model = onnxoptimizer.optimize(
            onnx.load(path_tmp),
            ["extract_constant_to_initializer",
             "eliminate_unused_initializer"])
        sess = ort.InferenceSession(optimized_model.SerializeToString())
        output_onnx[node] = torch.tensor(sess.run(
            None, {sess.get_inputs()[0].name: dummy_input})[0])

    print('Inferencing the pytorch model')
    output_pytorch = pytorch_model(
        torch.tensor(dummy_input), return_all_nodes=True)

    for k in output_pytorch:
        if k == sess.get_inputs()[0].name:
            continue
        print(k, output_onnx[k].shape)
        close = torch.allclose(output_onnx[k], output_pytorch[k])
        print('  close?', close)
        if not close:
            print('  max error', (output_onnx[k] - output_pytorch[k]).abs().max())

    import pdb; pdb.set_trace()


def load_model(weights_loaded=True):
    """
    Load the model architectures and weights
    """

    assert arguments.Config["model"]["name"] is None or arguments.Config["model"]["onnx_path"] is None, (
        "Conflict detected! User should specify model path by either --model or --onnx_path! "
        "The cannot be both specified.")

    assert arguments.Config["model"]["name"] is not None or arguments.Config["model"]["onnx_path"] is not None, (
        "No model is loaded, please set --model <modelname> for pytorch model or --onnx_path <filename> for onnx model.")

    if arguments.Config['model']['name'] is not None:
        # You can customize this function to load your own model based on model name.
        try:
            model_ori = eval(arguments.Config['model']['name'])()  # pylint: disable=eval-used
        except Exception:  # pylint: disable=broad-except
            print(f'Cannot load pytorch model definition "{arguments.Config["model"]["name"]}()". '
                  f'"{arguments.Config["model"]["name"]}()" must be a callable that returns a torch.nn.Module object.')
            import traceback
            traceback.print_exc()
            exit()
        model_ori.eval()

        if not weights_loaded:
            return model_ori

        if arguments.Config["model"]["path"] is not None:
            # Load pytorch model
            # You can customize this function to load your own model based on model name.
            sd = torch.load(expand_path(arguments.Config["model"]["path"]),
                            map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format, please modify model loader yourself.")
            try:
                model_ori.load_state_dict(sd)
            except RuntimeError:
                print('Failed to load the model')
                print('Keys in the state_dict of model_ori:')
                print(list(model_ori.state_dict().keys()))
                print('Keys in the state_dict trying to load:')
                print(list(sd.keys()))
                raise

    elif arguments.Config["model"]["onnx_path"] is not None:
        # Load onnx model
        model_ori, _ = load_model_onnx(expand_path(
            arguments.Config["model"]["onnx_path"]))

    else:
        print("Warning: pretrained model path is not given!")

    print(model_ori)
    print('Parameters:')
    for p in model_ori.named_parameters():
        print(f'  {p[0]}: shape {p[1].shape}')

    return model_ori

"""
This file shows how to use customized models and customized dataloaders.

Use the example configuration:
python abcrown.py --config exp_configs/tutorial_examples/custom_model_data_example.yaml
"""

import os
import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
import arguments



def try_model_fsb(in_dim=784, out_dim=10):
    model = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def simple_conv_model(in_channel, out_dim):
    """Simple Convolutional model."""
    model = nn.Sequential(
        nn.Conv2d(in_channel, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, out_dim)
    )
    return model


def two_relu_toy_model(in_dim=2, out_dim=2):
    """A very simple model, 2 inputs, 2 ReLUs, 2 outputs"""
    model = nn.Sequential(
        nn.Linear(in_dim, 2),
        nn.ReLU(),
        nn.Linear(2, out_dim)
    )
    # [relu(x+2y)-relu(2x+y)+2, 0*relu(2x-y)+0*relu(-x+y)]
    model[0].weight.data = torch.tensor([[1., 2.], [2., 1.]])
    model[0].bias.data = torch.tensor([0., 0.])
    model[2].weight.data = torch.tensor([[1., -1.], [0., 0.]])
    model[2].bias.data = torch.tensor([2., 0.])
    return model


def simple_box_data(spec):
    """a customized box data: x=[-1, 1], y=[-1, 1]"""
    eps = spec["epsilon"]
    if eps is None:
        eps = 2.
    X = torch.tensor([[0., 0.]]).float()
    labels = torch.tensor([0]).long()
    eps_temp = torch.tensor(eps).reshape(1, -1)
    data_max = torch.tensor(10.).reshape(1, -1)
    data_min = torch.tensor(-10.).reshape(1, -1)
    return X, labels, data_max, data_min, eps_temp

def all_node_split_test_model(in_dim=2, out_dim=2):
    """A very simple model, 2 inputs, 20 ReLUs, 2 outputs"""
    model = nn.Sequential(
        nn.Linear(in_dim, 20),
        nn.ReLU(),
        nn.Linear(20, out_dim)
    )
    model[0].weight.data = torch.tensor([[-1.1258, -1.1524],
                                        [ 0.2506, -0.4339],
                                        [ 0.8487,  0.6920],
                                        [-0.3160, -2.1152],
                                        [ 0.4681, -0.1577],
                                        [ 1.4437,  0.2660],
                                        [ 0.1665,  0.8744],
                                        [-0.1435, -0.1116],
                                        [ 0.4736, -0.0729],
                                        [-0.8460,  0.1241],
                                        [ 0.2664,  0.4124],
                                        [-1.1480, -0.9625],
                                        [ 0.2343,  0.1264],
                                        [ 0.6591, -1.6591],
                                        [-1.0093, -1.4070],
                                        [ 0.2204, -0.1970],
                                        [-1.0683, -0.0390],
                                        [ 0.6933, -0.0684],
                                        [-0.5896,  0.7262],
                                        [ 0.8356, -0.1248]])
    model[0].bias.data = torch.tensor([-0.0043,  0.0017,  0.0020, -0.0005, -0.0030,  0.0011, -0.0029, -0.0023,  0.0037,  0.0023, -0.0025,  0.0041, -0.0082, -0.0077,  0.0006, -0.0022, -0.0045,  0.0003, -0.0033,  0.0020])
    model[2].weight.data = torch.tensor([[ 1.2026, -1.0299, -0.0809,  0.4990, -0.6472, -0.2247,  0.0726, -0.2912, -0.5695,  0.8674,
                                        -0.6774,  0.2767,  0.1709, -0.2701, -0.5633,  0.2803, -1.0325, -0.6330,  0.3569, -0.0638],
                                        [ 0.0129,  0.2553, -0.2982, -0.1459, -0.1255,  0.1057, -0.9055,  0.4570,  0.4074,  0.3204,
                                        -0.0127,  0.7773, -0.0831,  0.3661, -0.6250, -0.7922, -0.1339,  0.2914,  0.2083, -0.4933]])
    model[2].bias.data = torch.tensor([ 0.2484,  0.4397])
    return model


def box_data(dim, low=0., high=1., segments=10, num_classes=10, eps=None):
    """Generate fake datapoints."""
    step = (high - low) / segments
    data_min = torch.linspace(low, high - step, segments).unsqueeze(1).expand(segments, dim)  # Per element lower bounds.
    data_max = torch.linspace(low + step, high, segments).unsqueeze(1).expand(segments, dim)  # Per element upper bounds.
    X = (data_min + data_max) / 2.  # Fake data.
    labels = torch.remainder(torch.arange(0, segments, dtype=torch.int64), num_classes)  # Fake label.
    eps = None  # Lp norm perturbation epsilon. Not used, since we will return per-element min and max.
    return X, labels, data_max, data_min, eps


def cifar10(spec, use_bounds=False):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    if use_bounds:
        # Option 1: for each example, we return its element-wise lower and upper bounds.
        # If you use this option, set --spec_type ("specifications"->"type" in config) to 'bound'.
        absolute_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        absolute_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        # Be careful with normalization.
        new_eps = torch.reshape(eps / std, (1, -1, 1, 1))
        data_max = torch.min(X + new_eps, absolute_max)
        data_min = torch.max(X - new_eps, absolute_min)
        # In this case, the epsilon does not matter here.
        ret_eps = None
    else:
        # Option 2: return a single epsilon for all data examples, as well as clipping lower and upper bounds.
        # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        if eps is None:
            raise ValueError('You must specify an epsilon')
        # Rescale epsilon.
        ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps


def simple_cifar10(spec):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True,\
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data,\
            batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
    if eps is None:
        raise ValueError('You must specify an epsilon')
    # Rescale epsilon.
    ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps



import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yfinance as yf
from pandas_datareader import data as web
import arguments  # α,β‐CROWN’s global config dictionary

class AssetNetMLP(nn.Module):
    def __init__(self, input_dim=50*4, hidden_dim=64, output_dim=4):
        """
        - input_dim = 50 days * 4 assets = 200
        - hidden_dim = 64 (arbitrary)
        - output_dim = 4 (one weight per asset)
        """
        super(AssetNetMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 50, 4)
        returns: Tensor of shape (batch_size, 4)
        """
        # Flatten the last two dims: (batch_size, 200)
        x_flat = x.view(x.size(0), -1)
        h = self.relu(self.fc1(x_flat))
        out = self.fc2(h)             # (batch_size, 4)
        weights = self.softmax(out)   # normalized allocations
        return weights

def my_model(in_channel, out_dim):
    """Simple Convolutional model."""
    model = AssetNetMLP()
    return model


from torch.nn import functional as F

class CustomSoftmax(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.exp(x)
        return e / (e.sum(dim=self.dim, keepdim=True) + self.eps)


class SimpleAssetAllocationModel(nn.Module):
    def __init__(self, input_channels=4, output_channels=4):
        super(SimpleAssetAllocationModel, self).__init__()
        
        input_size = input_channels
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_channels)
        self.softmax = CustomSoftmax(dim=-1)

        # self.fc4 = nn.Linear(4*50, 4)

    def forward(self, x):

        x = torch.flatten(x, start_dim=-2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)    

class FinalCNN(nn.Module):
    def __init__(self, input_channels=4, time_steps=50, hidden_size=100, dropout_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * time_steps, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_channels)
        self.softmax = CustomSoftmax(dim=-1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        # print(x,  self.softmax(x), self.softmax(x/10))
        
        return self.softmax(x)
    
def cnn_model():
    return FinalCNN()

# def lstm_model():
#     return FinalCNN()

# -----------------------------
# 1) Window‐creation & standardization code
# -----------------------------
def create_windows(data, indices, lookback, horizon):
    """
    data: pd.DataFrame with shape (T, n_assets)
    indices: list or Index of timestamps
    lookback: how many timesteps to “look back” for inputs
    horizon: how many timesteps to predict (unused for verification, but needed here)
    """
    X, y = [], []
    index_list = []

    for i in range(len(indices) - lookback - horizon):
        window_start = indices[i]
        input_end = indices[i + lookback]
        output_end = indices[i + lookback + horizon]

        # x_window: (lookback, n_assets)
        x_window = data.loc[window_start:input_end].iloc[:-1].values
        # y_window: (horizon, n_assets)
        y_window = data.loc[input_end:output_end].iloc[:horizon].values
        
        X.append(x_window)
        y.append(y_window)
        index_list.append(input_end)

    return np.array(X), np.array(y), index_list

def standardize(x, mean, std):
    # x: Tensor of shape (N, lookback, n_assets)
    # mean/std: Tensor of shape (n_assets,)
    return (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)

def get_asset_dataset(start, end):
    """
    Download price & VIX data from FRED/yfinance, compute returns.
    Returns:
      - returns_df: pd.DataFrame of pct_change returns from (“start” to “end”)
      - close_df:   pd.DataFrame of actual prices (aligned to returns_df index + VIX)
    """
    etfs = ['AGG', 'DBC', 'VTI']
    asset_names = etfs + ["VIX"]
    n_assets = len(asset_names)

    # Fetch VIX from FRED:
    vix = web.DataReader('VIXCLS', 'fred', start=start, end=end)
    # Fetch ETF closes:
    close_df = yf.download(etfs, start=start, end=end)["Close"]
    close_df["VIX"] = vix

    returns_df = close_df.pct_change().iloc[1:]
    close_df  = close_df.iloc[1:]
    return returns_df, close_df

# -----------------------------
# 2) Define train/val/test indices & compute windows
# -----------------------------
trainval_start = "2006-04-02"
trainval_end   = "2019-12-31"
test_start     = "2020-01-01"
test_end       = "2022-12-31"

returns_df, close_df = get_asset_dataset(trainval_start, test_end)

asset_names = list(returns_df.columns)
n_assets    = len(asset_names)  # should be 4 (AGG, DBC, VTI, VIX)

# Build train/val/test index masks
trainval_mask = (close_df.index >= trainval_start) & (close_df.index <= trainval_end)
trainval_indices = close_df[trainval_mask].index

val_size      = int(0.20 * len(trainval_indices))
train_indices = trainval_indices[:-val_size]
val_indices   = trainval_indices[-val_size:]

test_mask    = (close_df.index >= test_start) & (close_df.index <= test_end)
test_indices = close_df[test_mask].index

# Window parameters
lookback       = 50   # input length (50 days)
horizon        = 5    # (unused in verification, but needed to build windows)
allocation_step = 5   # (unused for now)

# Create windows for train / val / test
X_train, y_train, train_idx = create_windows(returns_df, train_indices, lookback, horizon)
X_val,   y_val,   val_idx   = create_windows(returns_df, val_indices,   lookback, horizon)
X_test,  y_test,  test_idx  = create_windows(returns_df, test_indices,  lookback, horizon)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)

# Compute global mean/std over train set (for standardization)
num_samples, seq_len, num_assets = X_train_tensor.shape
flattened = X_train_tensor.reshape(-1, num_assets)   # shape (num_samples*seq_len, num_assets)

mean = flattened.mean(dim=0)   # shape (n_assets,)
std  = flattened.std(dim=0)    # shape (n_assets,)
std[std == 0] = 1.0            # avoid zero‐division

# Standardize all splits
X_train_tensor = standardize(X_train_tensor, mean, std)
X_val_tensor   = standardize(X_val_tensor,   mean, std)
X_test_tensor  = standardize(X_test_tensor,  mean, std)

# Optionally save mean/std so you can “de‐standardize” later
saved_mean = mean
saved_std  = std

# -----------------------------
# 3) New dataset function for α,β‐CROWN
# -----------------------------
def asset_dataset(spec):
    """
    α,β‐CROWN will call this if your YAML has `data: { dataset: asset_dataset }`.
    spec is the “specification” dict (e.g. {"norm":".inf","epsilon":0.1}).

    We do:
      1. X = X_test_tensor  (shape: N x 50 x n_assets)
      2. labels = a dummy Tensor of ints (length N)
      3. data_max / data_min = element‐wise clipping bounds = X ± epsilon
      4. ret_eps = a 1‐element tensor containing epsilon

    Returns:
      X:        FloatTensor (N, 50, n_assets)
      labels:   LongTensor (N,)        # α,β‐CROWN doesn’t really use labels for bounding
      data_max: FloatTensor (N, 50, n_assets)
      data_min: FloatTensor (N, 50, n_assets)
      ret_eps:  FloatTensor of shape (1,) representing epsilon
    """

    i_start = arguments.Config["data"]["start"]
    i_end   = arguments.Config["data"]["end"]
    # 1) pick the test split
    X = X_test_tensor[::5][i_start:i_end]
    y = y_test_tensor[::5][i_start:i_end]
    N = X.shape[0]

    print(arguments.Config["model"]["path"])
    print(arguments.Config["model"]["name"])
    model = load_model()
    clean_output = model(X).detach()

    print(y.shape)

    MR = clean_output.unsqueeze(1) * y  # shape (N, 5, 4)

    MR = MR.view(N, 20)



    # 2) dummy labels (just zeros—verifier only needs a placeholder)
    # labels = torch.zeros(N, dtype=torch.long)+1

    # 3) compute epsilon (broadcastable)
    eps = spec.get("epsilon", 0.0)
    eps_tensor = torch.tensor(eps)

    # Build per‐element bounds: data_min = X - eps; data_max = X + eps
    data_min = X - eps_tensor
    data_max = X + eps_tensor

    combined = torch.cat((clean_output, MR), dim=1)

    # y = y[:, 0:1, :].squeeze(1)
    # print(X.shape, y.shape)

    return X, combined, data_max, data_min, eps_tensor
