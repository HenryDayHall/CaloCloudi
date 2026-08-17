# Consistency distillation of a trained teacher diffusion model.
#
# Start a new distillation;
#   python scripts/train_student.py <config.yaml> <teacher_checkpoint.pt>
# Resume a distillation;
#   python scripts/train_student.py <student_model.pt>
import sys
import time
import torch
import k_diffusion

from torch.nn.utils import clip_grad_norm_

from src.training.student import init_from_pretrained, init_from_scratch

user_input = sys.argv[1]

if user_input.endswith(".yaml"):
    if len(sys.argv) < 3:
        raise SystemExit(
            "Starting from a config also needs a teacher checkpoint;\n"
            "python scripts/train_student.py <config.yaml> <teacher_checkpoint.pt>"
        )
    config_path = user_input
    teacher_model_path = sys.argv[2]
    setup = init_from_scratch(config_path, teacher_model_path)
else:
    model_path = user_input
    setup = init_from_pretrained(model_path)

checkpointable = {
    name: setup[name]
    for name in ["model", "ema_model", "teacher_model", "optimiser"]
}

model = setup["model"]
ema_model = setup["ema_model"]  # the target model, an EMA of the online model
teacher_model = setup["teacher_model"]
optimiser = setup["optimiser"]
logger = setup["logger"]
config = logger.config
batch_size = config["training"]["batch_size"]
device = config["device"]
dtype = getattr(torch, config["training"]["dtype"])
torch.set_default_dtype(dtype)

# EMA decay of the target model; necessary for consistency distillation,
# not the same as an EMA decay of the online model itself
ema_decay = config["training"]["start_ema"]
logger.add_text(f"Target (ema) model decay rate {ema_decay}")
logger.add_text("No learning rate scheduler is used for distillation")

if len(logger.values["epoch_number"]) == 0:
    start_epoch = 0
    n_events = 0
    n_updates = 0
    logger.add_step(
        loss=0.0,
        ema_decay=ema_decay,
        n_events=n_events,
        n_updates=n_updates,
        epoch_number=start_epoch,
        time=time.time(),
    )
    logger.add_text("Distilling from scratch > creating initial checkpoint")
    logger.checkpoint_model(**checkpointable)
else:
    start_epoch = logger.values["epoch_number"][-1]
    n_events = logger.values["n_events"][-1]
    n_updates = logger.values["n_updates"][-1]
    logger.add_text(
        f"Distilling from checkpoint, after {n_events} events and {n_updates} updates"
    )

last_checkpoint_time = time.time()

end_epoch = config["training"]["end_epoch"]

logger.add_text(f"Distilling from epoch {start_epoch + 1} to {end_epoch}")
logger.do_validation(model)

checkpoint_time_interval_seconds = (
    config["training"]["checkpoint_time_interval_min"] * 60
)
checkpoint_batch_interval = config["training"]["checkpoint_batch_interval"]
cond_keys = config["model"]["cond_features"]

for epoch in range(start_epoch + 1, end_epoch + 1):
    logger.add_text(f"Distilling epoch {epoch}/{end_epoch}")
    # the validation steps may put the models into other modes
    model.train().requires_grad_(True)
    # the target model needs to be in the same mode as the online model,
    # but never requires gradients
    ema_model.train().requires_grad_(False)
    teacher_model.eval().requires_grad_(False)
    for batch in setup["dataloader"]:
        # each batch will have points, incident_energy, incident_direction
        n_events += batch_size
        n_updates += 1
        batch_cond = torch.cat(
            [batch[key] for key in cond_keys],
            dim=-1,
        )
        batch_cond = batch_cond.to(device, dtype=dtype)
        batch_cond = setup["preprocess_conditioning"].forward(batch_cond)
        batch_features = batch["points"][0].to(device, dtype=dtype)
        batch_features = setup["preprocess_features"].forward(batch_features)

        optimiser.zero_grad()
        # the noise and time steps are drawn inside the consistency loss
        loss = model.get_cd_loss(
            batch_features, batch_cond, teacher_model, ema_model, config
        )
        loss.backward()
        grad_norm = clip_grad_norm_(
            model.parameters(), config["training"]["max_grad_norm"]
        )
        optimiser.step()

        # the target model tracks the online model with a constant decay
        k_diffusion.utils.ema_update(model, ema_model, ema_decay)

        time_now = time.time()
        logger.add_step(
            loss=loss.item(),
            ema_decay=ema_decay,
            grad_norm=grad_norm,
            n_events=n_events,
            n_updates=n_updates,
            epoch_number=epoch,
            time=time_now,
        )
        if n_updates % 100 == 0:
            logger.save()
        if n_updates % config["training"]["validation_interval"] == 0:
            logger.do_validation(model)
            logger.save()

        if checkpoint_time_interval_seconds > 0:
            if time_now - last_checkpoint_time > checkpoint_time_interval_seconds:
                logger.do_validation(model)
                logger.checkpoint_model(**checkpointable)
                logger.save()
                last_checkpoint_time = time_now
        if checkpoint_batch_interval > 0:
            if n_updates % checkpoint_batch_interval == 0:
                logger.do_validation(model)
                logger.checkpoint_model(**checkpointable)
                logger.save()
    logger.checkpoint_model(**checkpointable)
    logger.do_validation(model)
    logger.save()
    last_checkpoint_time = time.time()
logger.add_text("Distillation complete")
