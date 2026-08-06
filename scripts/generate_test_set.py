import sys
import os
import numpy as np
from src.evaluation import inference

model_path = sys.argv[1]
model_name = os.path.basename(model_path)
model_dir = os.path.dirname(model_path)
output_path = os.path.join(model_dir, model_name.split(".")[0] + "_testset.npy")

sampler = inference.Sampler.from_model_path(model_path)
if True:
    n_events = 10
    print(f"Sampling {n_events} events")
    cond, points, target, sample = sampler.sample_from_dataset(
        "test", return_target=True, total_size=10
    )
    del cond
    del points
    print(f"Sample shape: {sample.shape}")

    np.save(output_path, sample)

    print("Getting points per layer from target")
    points_per_layer = inference.points_per_layer_from_target(target, sampler.config)
    # don't need the target anymore
    del target
    print(f"Max points per layer: {np.max(points_per_layer)}")

    print("Converting to physical points")
    physical_points, point_layer_ids = inference.sample_to_physical(
        sample, points_per_layer, sampler.config
    )
    np.save(output_path.replace(".npy", "_physical.npy"), physical_points)

   # print("Converting to cells")
   # cells = inference.sample_to_cells(physical_points, point_layer_ids, sampler.config)
   # np.save(output_path.replace(".npy", "_cells.npy"), cells)
   # print(f"Cell shape: {cells.shape}")

    print("Done")
