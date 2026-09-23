import h5py
import numpy as np
import sys

do_em = "EM" in sys.argv[1]

allshowers_padded = (
    "/data/dust/user/dayhallh/data/AllShowers/padded_all/Padded_All_{}_of_200.h5"
)
EM_padded = "/data/dust/user/dayhallh/data/AllShowers/padded_EM/Padded_EM_{}_of_200.h5"

if do_em:
    print("Doing electromagnetics")

    for i in range(0, 200):
        print(f"{i/200:.0%}", end="\r")
        with h5py.File(EM_padded.format(i), "w") as outputfile:
            with h5py.File(allshowers_padded.format(i), "r") as f:
                pdg = f["pdg"][:]
                mask = (pdg == 11) | (pdg == 22) | (pdg == -11)
                for key in f.keys():
                    outputfile.create_dataset(key, data=f[key][mask])

do_cc3 = "CC3" in sys.argv[1]

caloclouds3 = "/data/dust/group/ilc/sft-ml/datasets/sim-E1261AT600AP180-180/sim-E1261AT600AP180-180_file_{}.slcio.hdf5"
caloclouds3_reprocessed = "/data/dust/user/dayhallh/data/AllShowers/caloclouds3_matched/CaloClouds3_{}_of_90.h5"
if do_cc3:
    print("Doing CaloClouds3")

    id_reached = 0
    for i in range(0, 90):
        print(f"{i/90:.0%}", end="\r")
        with h5py.File(caloclouds3_reprocessed.format(i), "w") as outputfile:
            with h5py.File(caloclouds3.format(i), "r") as f:
                showers = f["events"][:]
                showers = np.moveaxis(showers, -1, -2)
                point_energies = showers[:, :, 3]
                order = np.argsort(-point_energies, axis=1)
                showers = showers[np.arange(showers.shape[0])[:, None], order, :]
                showers = showers[:, :6016]
                showers[:, :, 0] += 0.5
                showers[:, :, 1] += 1.0
                outputfile.create_dataset("showers", data=showers)
                outputfile.create_dataset("genE", data=f["energy"][:])
                pdgs = np.full(showers.shape[0], 22)
                outputfile.create_dataset("pdg", data=pdgs)
                outputfile.create_dataset("directions", data=np.array(f["p_norm_local"][:]).T)
                shower_ids = np.arange(showers.shape[0]) + id_reached
                id_reached += showers.shape[0]
                outputfile.create_dataset("shower_ids", data=shower_ids)



print("\n DONE")
