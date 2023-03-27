from usdf.usdf_gaussian.training import Trainer
from usdf.usdf_gaussian.generation import Generator


def get_model(cfg, dataset, device=None):
    return None


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, generation_cfg, device=None):
    generator = Generator(cfg, model, generation_cfg, device)
    return generator
