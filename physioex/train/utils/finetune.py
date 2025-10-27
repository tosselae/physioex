import importlib
from typing import Dict, Type, Union

from physioex.train.models import load_model
from physioex.train.networks.base import SleepModule
from physioex.train.utils.train import train

# the finetune function is similar to the train function but it has a few differences:
# - models can't be finetuned from scratch, they must be loaded from a checkpoint, or being a pretrained model from physioex.models
# tipycally when fine-tuning a model you want to setup the learning rate


def finetune(
    model: Union[
        SleepModule, str
    ] = None,  # if passed model_class, model_config and resume are ignored
    model_class: Type[SleepModule] = None,
    model_config: Dict = None,
    model_checkpoint: str = None,
    learning_rate: float = 1e-7,  # if None not updated
    weight_decay: Union[str, float] = "auto",  # if None not updated
    train_kwargs: Dict = None,
) -> str:
    
    if (model is None):
        assert (model_class is None or model_config is None or model_checkpoint is None
        ), "Either model or model_class, model_config and model_checkpoint must be passed"

    if isinstance(model, str) and ":" in model:
        # import the module and the class
        module, class_name = model.split(":")
        model_class = getattr(importlib.import_module(module), class_name)

    if not isinstance(model, SleepModule):
        model = load_model(
            model=model_class,
            model_kwargs=model_config,
            ckpt_path=model_checkpoint,
        ).train()

    model.learning_rate = learning_rate
    model.weight_decay = weight_decay if weight_decay != "auto" else learning_rate

    model.configure_optimizers()

    train_kwargs["resume"] = False

    return train(model=model, **train_kwargs)
