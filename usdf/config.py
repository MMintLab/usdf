from torchvision import transforms

method_dict = {}


def get_model(cfg, dataset, device=None):
    """
    Args:
    - cfg (dict): training config.
    - dataset (dataset): training dataset (in case model depends).
    - device (device): pytorch device.
    """
    method = cfg['method']
    model = method_dict[method].config.get_model(cfg, dataset, device=device)
    return model


def get_trainer(cfg, model, device=None):
    """
    Return trainer instance.

    Args:
    - cfg (dict): training config
    - model (nn.Module): model which is used
    - device (device): pytorch device
    """
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, generation_cfg, device=None):
    """
    Return generator instance.

    Args:
    - cfg (dict): configuration dict
    - model (nn.Module): model which is used
    - generation_cfg (dict): generation configuration dict
    - device (torch.device): pytorch device
    """
    method = cfg['method']
    generator = method_dict[method].config.get_generator(cfg, model, generation_cfg, device)
    return generator


def get_dataset(mode, cfg):
    """
    Args:
    - mode (str): dataset mode [train, val, test].
    - cfg (dict): training config.
    """
    dataset_type = cfg['data'][mode]['dataset']

    # Build dataset transforms.
    transforms_ = get_transforms(cfg)

    raise Exception("Unknown requested dataset type: %s" % dataset_type)

    return dataset


def get_transforms(cfg):
    transforms_info = cfg['data'].get('transforms')
    if transforms_info is None:
        return None

    transform_list = []
    for transform_info in transforms_info:
        transform_type = transform_info["type"]
        transform = None

        raise Exception("Unknown transform type: %s" % transform_type)

        transform_list.append(transform)

    composed = transforms.Compose(transform_list)
    return composed
