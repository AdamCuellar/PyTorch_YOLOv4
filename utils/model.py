import math
import os
import numpy as np
import torch
import torch.nn as nn
from utils import torch_utils
from utils.layers import MixConv2d, DeformConv2d, FeatureConcat, FeatureConcat2, FeatureConcat3, FeatureConcat_l, Swish, \
    WeightedFeatureFusion, YOLOLayer, Mish

def create_modules(module_defs, img_size=None):
    # Constructs module list of layer blocks from module configuration in module_defs

    netParams = module_defs.pop(0)
    img_size = [netParams["height"], netParams["width"]] if img_size is None else img_size
    output_filters = [netParams["channels"]]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize'] if "batch_normalize" in mdef else 0
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'deformableconvolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('DeformConv2d', DeformConv2d(output_filters[-1],
                                                       filters,
                                                       kernel_size=k,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       stride=stride,
                                                       bias=not bn,
                                                       modulation=True))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', nn.Mish())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'route2':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat2(layers=layers)

        elif mdef['type'] == 'route3':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat3(layers=layers)

        elif mdef['type'] == 'route_lhalf':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])//2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            netParams["resize"] = 1-mdef["resize"] if "resize" in mdef else 0
            netParams["jitter"] = mdef["jitter"] if "jitter" in mdef else 0
            yolo_index += 1
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                scale_x_y=mdef["scale_x_y"],
                                new_coords=mdef["new_coords"] if "new_coords" in mdef else 0,
                                iou_loss=mdef["iou_loss"],
                                iou_thresh=mdef["iou_thresh"],
                                iou_normalizer=mdef["iou_normalizer"] if "iou_normalizer" in mdef else 1.,
                                cls_normalizer=mdef["cls_normalizer"] if "cls_normalizer" in mdef else 1.,
                                obj_normalizer=mdef["obj_normalizer"] if "obj_normalizer" in mdef else 1.)


            # # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            # try:
            #     j = layers[yolo_index] if 'from' in mdef else -1
            #     bias_ = module_list[j][0].bias  # shape(255,)
            #     bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
            #     #bias[:, 4] += -4.5  # obj
            #     bias[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)  # obj (8 objects per 640 image)
            #     bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
            #     module_list[j][0].bias = nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            # except:
            #     print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return netParams, module_list, routs_binary

class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, module_defs, img_size=None, verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = module_defs
        self.netParams, self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose)  # print model description

    def forward(self, x, verbose=False, return_strides=False, study_preds=False):
        b, c, h, w = x.shape  # height, width
        yolo_out, out = [], []
        strides = []
        if verbose:
            print('0', x.shape)
            str = ''

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l']:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                strides.append(h//x.shape[2])
                yolo_out.append(module(x, out, study_preds))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if return_strides:
            return strides

        if self.training:  # train
            return yolo_out

        if study_preds: # inference or test but cat
            x, p = zip(*yolo_out)
            return x, p
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info()

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)

    def load_darknet_weights(self, weights):
        # Parses and loads the weights stored in 'weights'

        # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
        fileName, ext = os.path.splitext(weights)
        cutoff = -1
        if is_int(ext):
            cutoff = int(ext)

        # Read weights file
        with open(weights, 'rb') as f:
            # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
            self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
            self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

            weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

        ptr = 0
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv = module[0]
                if mdef['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn = module[1]
                    nb = bn.bias.numel()  # number of biases
                    # Bias
                    bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                    ptr += nb
                    # Weight
                    bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                    ptr += nb
                    # Running Mean
                    bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                    ptr += nb
                    # Running Var
                    bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                    ptr += nb
                else:
                    # Load conv. bias
                    nb = conv.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                    conv.bias.data.copy_(conv_b)
                    ptr += nb
                # Load conv. weights
                nw = conv.weight.numel()  # number of weights
                conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
                ptr += nw

def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']
