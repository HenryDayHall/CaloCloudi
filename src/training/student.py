import torch
import os
from . import utils
from ..diffusion import Diffusion
from ..data.transforms import preprocessing
from ..data.dataset import from_config as dataset_from_config
from ..evaluation.inference import evaluating


class ValidationChecker:
    batch_size = 32
    # the consistency loss draws its own noise and time steps internally,
    # so a fixed seed is used to keep successive validations comparable
    seed = 1

    def __init__(
        self,
        config,
        preprocess_conditioning,
        preprocess_features,
        teacher_model,
        target_model,
        validation_size=1_000,
    ):
        self.config = config
        self.device = config["device"]
        self.dtype = getattr(torch, config["training"]["dtype"])
        self.n_batches = validation_size // self.batch_size
        # both models are updated in place during training,
        # so these references stay current
        self.teacher_model = teacher_model
        self.target_model = target_model
        dataset = dataset_from_config(config, dataset_part="val")
        dataset.bs = min(len(dataset), validation_size)
        all_val = dataset[0]
        target = all_val["points"]
        cond = [
            torch.from_numpy(all_val[name]).to(self.device, dtype=self.dtype)
            for name in config["model"]["cond_features"]
        ]
        self.cond = preprocess_conditioning.forward(torch.cat(cond, dim=1))
        self.target = preprocess_features.forward(
            torch.from_numpy(target).to(self.device, dtype=self.dtype)
        )
        self.callables = {"loss": self.loss}

    def total_real_val_points(self):
        n_events = self.batch_size * self.n_batches
        return (self.target[:n_events, :, 3] > 0).sum()

    def loss(self, model):
        found = []
        rng_devices = []
        if torch.device(self.device).type == "cuda":
            rng_devices = [torch.device(self.device)]
        # all three models are put into eval mode; the online and target model
        # must be in the same mode for the consistency loss to be meaningful
        with evaluating(model), evaluating(self.target_model):
            with evaluating(self.teacher_model):
                with torch.no_grad():
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(self.seed)
                        for batch in range(self.n_batches):
                            start = batch * self.batch_size
                            end = (batch + 1) * self.batch_size
                            here = model.get_cd_loss(
                                self.target[start:end],
                                self.cond[start:end],
                                self.teacher_model,
                                self.target_model,
                                self.config,
                            )
                            found.append(here.detach())
        found = torch.stack(found).cpu().numpy()
        return found

    def get_info(self):
        return {
            "batch_size": self.batch_size,
            "total_real_points": self.total_real_val_points(),
        }


def common(config):
    dataloader = utils.get_dataloader(config)
    datatype = getattr(torch, config["training"]["dtype"])
    # the student ("online") model, the only network that is trained
    model = Diffusion(config, distillation=True)
    model.to(config["device"], dtype=datatype)
    # the target model used inside the consistency loss and for sampling
    # from the consistency model; updated as an EMA of the online model.
    # Called ema_model here so the checkpoint files match the teacher's naming.
    ema_model = Diffusion(config, distillation=True)
    ema_model.load_state_dict(model.state_dict())
    ema_model.to(config["device"], dtype=datatype)
    # the target model needs to be in the same mode as the online model,
    # but never requires gradients
    ema_model.train().requires_grad_(False)
    # the teacher model, used as the score function in the ODE solver
    teacher_model = Diffusion(config, distillation=False)
    teacher_model.to(config["device"], dtype=datatype)
    teacher_model.eval().requires_grad_(False)
    optimiser = utils.get_optimiser(config, model)
    preprocess_conditioning = preprocessing(config, "conditioning")
    preprocess_features = preprocessing(config, "features")
    validation_checker = ValidationChecker(
        config,
        preprocess_conditioning,
        preprocess_features,
        teacher_model,
        ema_model,
    )
    setup_dict = {
        "dataloader": dataloader,
        "model": model,
        "ema_model": ema_model,
        "teacher_model": teacher_model,
        "optimiser": optimiser,
        "preprocess_conditioning": preprocess_conditioning,
        "preprocess_features": preprocess_features,
    }
    return setup_dict, validation_checker


def init_from_scratch(config_path, teacher_model_path):
    logger = utils.Logger(config_path,
                          run_type="student",
                          run_information={"teacher_model_path": teacher_model_path})
    config = logger.config
    setup_dict, validation_checker = common(config)
    for name, function in validation_checker.callables.items():
        logger.add_validation_function(name, function)
    logger.add_text(
        "Info from validation checker:\n" + str(validation_checker.get_info())
    )
    teacher_state_dict = torch.load(teacher_model_path, map_location=config["device"])
    setup_dict["teacher_model"].load_state_dict(teacher_state_dict)
    logger.add_text(f"Teacher model loaded from {teacher_model_path}")
    if config["training"]["cm_random_init"]:
        logger.add_text(
            "cm_random_init > online and target (ema) models"
            " keep their random initialisation"
        )
    else:
        setup_dict["model"].load_state_dict(teacher_state_dict)
        setup_dict["ema_model"].load_state_dict(teacher_state_dict)
        logger.add_text("Online and target (ema) models initialised from the teacher")
    setup_dict["logger"] = logger
    return setup_dict


def init_from_pretrained(model_path):
    logger = utils.Logger.from_model_path(model_path, run_type="student")
    config = logger.config
    device = config["device"]
    setup_dict, validation_checker = common(config)

    for name, function in validation_checker.callables.items():
        logger.add_validation_function(name, function)

    setup_dict["model"].load_state_dict(torch.load(model_path, map_location=device))
    model_dir, model_base = os.path.split(model_path)

    ema_model_path = os.path.join(model_dir, model_base.replace("model", "ema_model"))
    setup_dict["ema_model"].load_state_dict(
        torch.load(ema_model_path, map_location=device)
    )

    teacher_model_path = os.path.join(
        model_dir, model_base.replace("model", "teacher_model")
    )
    setup_dict["teacher_model"].load_state_dict(
        torch.load(teacher_model_path, map_location=device)
    )

    optimiser_state_dict_path = os.path.join(
        model_dir, model_base.replace("model", "optimiser")
    )
    setup_dict["optimiser"].load_state_dict(
        torch.load(optimiser_state_dict_path, map_location=device)
    )
    setup_dict["logger"] = logger
    return setup_dict
