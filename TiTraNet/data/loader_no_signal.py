from torch.utils.data import DataLoader

from kigan.data.trajectories_no_signal import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim
        )

    loader = DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate,
        # pin_memory=True,
        # persistent_workers=True
        )
    return dset, loader
