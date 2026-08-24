import yaml
import sys
from src.data import read_write

conf_path = sys.argv[1]
conf = yaml.safe_load(open(conf_path, 'r'))
cond_cols = conf["model"]["cond_features"]
cond_on_disk = [conf["data"][f"{col}_key"] for col in cond_cols]
per_event, events = read_write.read_raw_regaxes(conf, "train", total_size=400, per_event_cols=cond_on_disk)
points = events[events[:, :, 3]>0]
import numpy as np
print(f"Mean {np.mean(points, axis=0)}")
print(f"Std {np.std(points, axis=0)}")
log_e = np.log(points[:, 3])
print(f"Log e mean {np.mean(log_e)}")
print(f"Log e std {np.std(log_e)}")
