import torch
import k_diffusion
from . import utils
from ..diffusion import Diffusion
from ..data.transforms import preprocessing


def common(config):
    dataloader = utils.get_dataloader(config)
    datatype = getattr(torch, config["training"]["dtype"])
    model = Diffusion(config, distillation=False)
    model.to(config["device"], dtype=datatype)
    ema_model = Diffusion(config, distillation=False)
    ema_model.load_state_dict(model.state_dict())
    ema_model.to(config["device"], dtype=datatype)
    ema_model.eval().requires_grad_(False)
    ema_sched = k_diffusion.utils.EMAWarmup(
        power=config["training"]["ema_power"],
        max_value=config["training"]["ema_max_value"],
    )
    optimiser = utils.get_optimiser(config, model)
    sample_density = utils.get_sample_density(config)
    preprocess_conditioning = preprocessing(config, "conditioning")
    preprocess_features = preprocessing(config, "features")
    setup_dict = {
        "dataloader": dataloader,
        "model": model,
        "ema_model": ema_model,
        "ema_sched": ema_sched,
        "optimiser": optimiser,
        "sample_density": sample_density,
        "preprocess_conditioning": preprocess_conditioning,
        "preprocess_features": preprocess_features,
    }
    return setup_dict


def init_from_scratch(config_path):
    logger = utils.Logger(config_path)
    config = logger.config
    setup_dict = common(config)
    scheduler = utils.get_scheduler(config, setup_dict["optimiser"], 0)
    setup_dict["logger"] = logger
    setup_dict["scheduler"] = scheduler
    return setup_dict


def init_from_pretrained(model_path):
    logger = utils.Logger.from_model_path(model_path)
    config = logger.config
    setup_dict = common(config)

    setup_dict["model"].load_state_dict(torch.load(model_path))

    ema_model_path = model_path.replace("model", "ema_model")
    setup_dict["ema_model"].ema_model.load_state_dict(torch.load(ema_model_path))
    setup_dict["ema_model"].eval().requires_grad_(False)

    ema_sched_path = model_path.replace("model", "ema_sched")
    setup_dict["ema_sched"].load_state_dict(torch.load(ema_sched_path))

    epoch_number = logger.values["epoch_number"][-1]
    optimiser_state_dict_path = model_path.replace("model", "optimiser")
    setup_dict["optimiser"].load_state_dict(torch.load(optimiser_state_dict_path))

    scheduler_state_dict_path = model_path.replace("model", "scheduler")
    scheduler = utils.get_scheduler(config, setup_dict["optimiser"], epoch_number)
    scheduler.load_state_dict(torch.load(scheduler_state_dict_path))
    setup_dict["logger"] = logger
    setup_dict["scheduler"] = scheduler
    return setup_dict
