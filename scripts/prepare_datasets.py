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
                outputfile.create_dataset("directions", data=f["p_norm_local"][:])
                shower_ids = np.arange(showers.shape[0]) + id_reached
                id_reached += showers.shape[0]
                outputfile.create_dataset("shower_ids", data=shower_ids)


do_merge = "merge" in sys.argv[1]
merge_file = "/data/dust/user/dayhallh/data/AllShowers/merged_cc3_asEM/merged_CC3_EM_{}_of_200.h5"

if do_merge:
    # start by working out how many in each file
    cc3_index_list = []
    for i in range(90):
        with h5py.File(caloclouds3_reprocessed.format(i), "r") as f:
            n_showers = f["showers"].shape[0]
            for j in range(n_showers):
                cc3_index_list.append((i, j))
    as_em_index_list = []
    for i in range(200):
        with h5py.File(EM_padded.format(i), "r") as f:
            n_showers = f["showers"].shape[0]
            for j in range(n_showers):
                as_em_index_list.append((i, j))
    # shuffle both lists
    np.random.shuffle(cc3_index_list)
    np.random.shuffle(as_em_index_list)
    
    total_files = 200
    batch_size_cc3 = int(np.ceil(len(cc3_index_list) / total_files))
    batch_size_em = int(np.ceil(len(as_em_index_list) / total_files))

    total_batch_size = batch_size_cc3 + batch_size_em
    print(f"Making batches of {total_batch_size}, with {batch_size_cc3} CC3 and {batch_size_em} EM")
    start_idx = 0

    for b in range(total_files):
        print(f"{b/total_files:.0%}", end="\r")

        cc3_idxs = cc3_index_list[b*batch_size_cc3:(b+1)*batch_size_cc3]
        as_em_idxs = as_em_index_list[b*batch_size_em:(b+1)*batch_size_em]
        random_batch_order = np.random.permutation(total_batch_size).tolist()
        showers = np.zeros((total_batch_size, 6016, 4), dtype=np.float32)
        pdgs = np.zeros(total_batch_size, dtype=np.int32)
        shower_ids = np.arange(total_batch_size) + start_idx
        start_idx += total_batch_size
        genE = np.zeros(total_batch_size, dtype=np.float32)
        directions = np.zeros((total_batch_size, 3), dtype=np.float32)
        cc3_shower_idx = -np.ones(total_batch_size, dtype=np.int32)
        as_em_shower_idx = -np.ones(total_batch_size, dtype=np.int32)
        for file_idx, idx in cc3_idxs:
            with h5py.File(caloclouds3_reprocessed.format(file_idx), "r") as f:
                store_idx = random_batch_order.pop()
                showers[store_idx] = f["showers"][idx]
                pdgs[store_idx] = f["pdg"][idx]
                directions[store_idx] = f["directions"][:].T[idx]
                genE[store_idx] = f["genE"][idx, 0]
                cc3_shower_idx[store_idx] = f["shower_ids"][idx]

        for file_idx, idx in as_em_idxs:
            with h5py.File(EM_padded.format(file_idx), "r") as f:
                store_idx = random_batch_order.pop()
                showers[store_idx] = f["showers"][idx]
                pdgs[store_idx] = f["pdg"][idx]
                directions[store_idx] = f["directions"][idx]
                import ipdb; ipdb.set_trace()
                genE[store_idx] = f["genE"][idx]
                as_em_shower_idx[store_idx] = f["shower_ids"][idx]

        with h5py.File(merge_file.format(b), "w") as outputfile:
            outputfile.create_dataset("showers", data=showers)
            outputfile.create_dataset("pdg", data=pdgs)
            outputfile.create_dataset("directions", data=directions)
            outputfile.create_dataset("shower_ids", data=shower_ids)
            outputfile.create_dataset("genE", data=genE)
            outputfile.create_dataset("cc3_shower_idx", data=cc3_shower_idx)
            outputfile.create_dataset("as_em_shower_idx", data=as_em_shower_idx)


print("\n DONE")
