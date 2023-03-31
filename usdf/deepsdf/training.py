import torch.utils.data
from usdf.training import BaseTrainer


class Trainer(BaseTrainer):

    def pretrain(self, *args, **kwargs):
        # Nothing to do.
        pass

    def train(self, train_dataset: torch.utils.data.Dataset):
        pass
