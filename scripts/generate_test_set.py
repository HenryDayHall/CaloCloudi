import sys
import os
import numpy as np
from src.evaluation import inference

model_path = sys.argv[1]
model_name = os.path.basename(model_path)
model_dir = os.path.dirname(model_path)
output_path = os.path.join(model_dir, model_name.split(".")[0] + "_testset.npy")

sampler = inference.Sampler.from_model_path(model_path)
cond, points, sample, _ = sampler.sample_from_dataset("test", return_target=False, total_size=10)

np.save(output_path, sample)
