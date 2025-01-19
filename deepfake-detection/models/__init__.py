from .clip_models import CLIPModel
from .fatformer import CLIPModel as FatFormer

VALID_NAMES = [
    "CLIP:RN50",
    "CLIP:RN101",
    "CLIP:RN50x4",
    "CLIP:RN50x16",
    "CLIP:RN50x64",
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",
    "CLIP:ViT-L/14@336px",
    "FAT:ViT-B/32",
    "FAT:ViT-B/16",
    "FAT:ViT-L/14",
]


def build_model(args):
    if args.backbone.startswith("CLIP:"):
        assert args.backbone in VALID_NAMES
        return CLIPModel(args.backbone[5:], args)
    else:
        raise NotImplementedError


def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("CLIP:"):
        return CLIPModel(name[5:])
    elif name.startswith("FAT:"):
        return FatFormer(name[4:])
    else:
        assert False
