from usdf.usdf.models.uncertain_sdf import USDF, Dist
from usdf.usdf.training import Trainer
from usdf.usdf.generation import Generator


def get_model(cfg, dataset, device=None):
    num_examples = len(dataset)

    num_components = cfg["model"].get("num_components", 1)
    dist_class = Dist(cfg["model"]["distribution"])

    model = USDF(num_examples, cfg["model"]["z_object_size"], cfg["model"]["use_encoder"], num_components, dist_class,
                 device).to(device)
    return model


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, generation_cfg, device=None):
    generator = Generator(cfg, model, generation_cfg, device)
    return generator
