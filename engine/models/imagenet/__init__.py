from torchvision.models import (
    alexnet,
    densenet121,
    googlenet,
    inception_v3,
    mnasnet1_0,
    mobilenet_v2,
    resnet50,
    squeezenet1_1,
    vgg16_bn,
    vgg19_bn,
)

try:
    from torchvision.models import regnet_x_1_6gf, resnext50_32x4d

    from .vision_transformer import vit_b_16
except:
    regnet_x_1_6gf = None
    resnext50_32x4d = None
    vit_b_16 = None
