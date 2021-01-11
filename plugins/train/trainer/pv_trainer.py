from ._base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)