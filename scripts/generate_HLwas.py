from src.evaluation import summarise
import glob
paths = glob.glob("/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/*/last_best_seen.txt")
bests = []
for path in paths:
    with open(path, 'r') as f:
        bests.append(f.read())
        
for path in bests:
    try:
        slicedHL = summarise.SlicedWassersteinHL.from_model_path(path)
    except Exception as e:
        print(path)
        print(e)
