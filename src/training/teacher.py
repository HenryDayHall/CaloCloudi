import torch
import k_diffusion
from . import utils
from ..diffusion import Diffusion
from ..data.transforms import preprocessing
from ..data import read_write


class ValidationChecker:
    batch_size = 32

    def __init__(
        self,
        config,
        sample_density,
        preprocess_conditioning,
        preprocess_features,
        validation_size=1_000,
    ):
        self.config = config
        self.device = config["device"]
        self.dtype = getattr(torch, config["training"]["dtype"])
        self.sample_density = sample_density
        self.n_batches = validation_size // self.batch_size
        cond_feature_names = config["model"]["cond_features"]
        names_in_data = [config["data"][f"{name}_key"] for name in cond_feature_names]
        cond, target = read_write.read_raw_regaxes(
            config, part="val", total_size=validation_size, per_event_cols=names_in_data
        )
        self.cond = preprocess_conditioning.forward(
            torch.from_numpy(cond).to(self.device, dtype=self.dtype)
        )
        self.target = preprocess_features.forward(
            torch.from_numpy(target).to(self.device, dtype=self.dtype)
        )
        self.noise = torch.randn_like(self.target).to(self.device, dtype=self.dtype)
        self.sigma = sample_density([self.target.shape[0]], device=self.device)
        self.callables = {"loss": self.loss}

    def total_real_val_points(self):
        n_events = self.batch_size * self.n_batches
        return (self.target[:n_events, :, 3] > 0).sum()

    def loss(self, model):
        model.eval()
        found = []
        with torch.no_grad():
            for batch in range(self.n_batches):
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                here = model.get_loss(
                    self.target[start:end],
                    self.noise[start:end],
                    self.sigma[start:end],
                    self.cond[start:end],
                )
                found.append(here.detach())
        model.train()
        found = torch.tensor(found).cpu().numpy()
        return found

    def get_info(self):
        return {
            "batch_size": self.batch_size,
            "total_real_points": self.total_real_val_points(),
        }


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
    validation_checker = ValidationChecker(
        config,
        sample_density,
        preprocess_conditioning,
        preprocess_features,
    )
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
    return setup_dict, validation_checker


def init_from_scratch(config_path):
    logger = utils.Logger(config_path)
    config = logger.config
    setup_dict, validation_checker = common(config)
    for name, function in validation_checker.callables.items():
        logger.add_validation_function(name, function)
    logger.add_text("Info from validation checker:\n" + validation_checker.get_info())
    scheduler = utils.get_scheduler(config, setup_dict["optimiser"], 0)
    setup_dict["logger"] = logger
    setup_dict["scheduler"] = scheduler
    return setup_dict


def init_from_pretrained(model_path):
    logger = utils.Logger.from_model_path(model_path)
    config = logger.config
    setup_dict, validation_functions_dict = common(config)

    for name, function in validation_functions_dict.items():
        logger.add_validation_function(name, function)

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
