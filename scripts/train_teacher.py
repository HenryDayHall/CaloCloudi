# TODO introduce a validation setup
import sys
import time
import torch
import k_diffusion

from torch.nn.utils import clip_grad_norm_

from src.training.teacher import init_from_pretrained, init_from_scratch

user_input = "/home/dayhallh/training/splitCC/CaloClouds_diffusion/config/default.yaml"
# user_input = sys.argv[1]
if user_input.endswith(".yaml"):
    config_path = user_input
    setup = init_from_scratch(config_path)
else:
    model_path = user_input
    setup = init_from_pretrained(model_path)

checkpointable = {
    name: setup[name]
    for name in ["model", "ema_model", "ema_sched", "optimiser", "scheduler"]
}

model = setup["model"]
model.train().requires_grad_(True)
logger = setup["logger"]
config = logger.config
batch_size = config["training"]["batch_size"]
device = config["device"]
dtype = getattr(torch, config["training"]["dtype"])
torch.set_default_dtype(dtype)

if len(logger.values["epoch_number"]) == 0:
    start_epoch = 0
    n_events = 0
    n_updates = 0
    logger.add_step(
        loss=0.0,
        ema_loss=0.0,
        n_events=n_events,
        n_updates=n_updates,
        epoch_number=start_epoch,
        time=time.time(),
    )
    logger.add_text("Training from scratch > creating initial checkpoint")
    logger.checkpoint_model(**checkpointable)
else:
    start_epoch = logger.values["epoch_number"][-1]
    n_events = logger.values["n_events"][-1]
    n_updates = logger.values["n_updates"][-1]
    logger.add_text(
        f"Training from checkpoint, after {n_events} events and {n_updates} updates"
    )

last_checkpoint_time = time.time()

end_epoch = config["training"]["end_epoch"]

logger.add_text(f"Training from epoch {start_epoch + 1} to {end_epoch}")

checkpoint_time_interval_seconds = (
    config["training"]["checkpoint_time_interval_min"] * 60
)
checkpoint_batch_interval = config["training"]["checkpoint_batch_interval"]

for epoch in range(start_epoch + 1, end_epoch + 1):
    logger.add_text(f"Training epoch {epoch}/{end_epoch}")
    # the validation steps may put the model into an eval mode
    model.train().requires_grad_(True)
    for batch in setup["dataloader"]:
        # each batch will have points, incident_energy, incident_direction
        n_events += batch_size
        n_updates += 1
        batch_cond = torch.cat(
            [batch["incident_energy"], batch["incident_direction"]],
            dim=-1,
        )
        batch_cond = batch_cond.to(device, dtype=dtype)
        batch_cond = setup["preprocess_conditioning"].forward(batch_cond)
        batch_features = batch["points"][0].to(device, dtype=dtype)
        batch_features = setup["preprocess_features"].forward(batch_features)
        noise = torch.randn_like(batch_features)
        sigma = setup["sample_density"](
            [batch_features.shape[0]], device=batch_features.device
        )

        loss = model.get_loss(batch_features, noise, sigma, batch_cond)
        loss.backward()
        grad_norm = clip_grad_norm_(
            model.parameters(), config["training"]["max_grad_norm"]
        )
        setup["optimiser"].step()
        setup["scheduler"].step()

        ema_decay = setup["ema_sched"].get_value()
        k_diffusion.utils.ema_update(model, setup["ema_model"], ema_decay)
        setup["ema_sched"].step()

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

        if checkpoint_time_interval_seconds > 0:
            if time_now - last_checkpoint_time > checkpoint_time_interval_seconds:
                logger.checkpoint_model(**checkpointable)
                logger.save()
                last_checkpoint_time = time_now
        if checkpoint_batch_interval > 0:
            if n_updates % checkpoint_batch_interval == 0:
                logger.checkpoint_model(**checkpointable)
                logger.save()
    logger.checkpoint_model(**checkpointable)
    logger.save()
    last_checkpoint_time = time.time()
logger.add_text("Training complete")
