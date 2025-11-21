from torch.utils.data import DataLoader

from kigan.data.trajectories import seq_collate
from kigan.data.trajectories_eval import TrajectoryDatasetEval   # 새로 만든 평가용 Dataset

def data_loader_eval(args, path):
    dset = TrajectoryDatasetEval(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        delim=args.delim,
        min_ped=1
    )
    loader = DataLoader(
        dset,
        batch_size=1,   # ✅ 파일 단위
        shuffle=False,  # ✅ 순서 보존
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate
    )
    return dset, loader