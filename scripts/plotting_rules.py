import glob
import os
import matplotlib

known_tags = [
        'padded photons v2',
        'generalist and moe all',
        'only moe all',
        'distill of padded photons v1',
        'only mos all',
        'distilled 2 from padded photons v1',
        'padded all v1',
        'generalist and mos all'
        ]

def tag_from_sample_path(path):
    run_dir = os.path.dirname(os.path.dirname(path))
    tag_glob = glob.glob(os.path.join(run_dir, "_a_*"))[0]
    tag = os.path.basename(tag_glob)[3:].replace("_", " ")
    return tag

log_dir = "/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs"

run_dirs = []
for tag in known_tags:
    run_dir = glob.glob(os.path.join(log_dir, f"*/_a_{tag.replace(' ', '_')}"))[0]
    run_dirs.append(os.path.basename(run_dir))


pretty_names = {
        "padded photons v2": "Only Photons",
        "padded all v1": "Simple Conditioning",
        "only mos all": "Only MoS",
        "only moe all": "Only MoE",
        "generalist and mos all": "Generalist and MoS",
        "generalist and moe all": "Generalist and MoE",
        "distill of padded photons v1": "Distilled Only Photons",
        "distilled 2 from padded photons v1": "Distilled Only Photons",
        "geant4": "GEANT4",
        "cc3": "CaloClouds 3",
        }

colours = {
        "padded photons v2": matplotlib.cm.tab10(0),
        "padded all v1": matplotlib.cm.tab10(1),
        "only mos all": matplotlib.cm.tab10(2),
        "only moe all": matplotlib.cm.tab10(5),
        "generalist and mos all": matplotlib.cm.tab10(4),
        "generalist and moe all": matplotlib.cm.tab10(3),
        "distill of padded photons v1": matplotlib.cm.tab10(6),
        "distilled 2 from padded photons v1": matplotlib.cm.tab10(8),  # must skip 7
        "geant4": matplotlib.cm.tab10(7),
        "cc3": matplotlib.cm.tab10(9),
        }

