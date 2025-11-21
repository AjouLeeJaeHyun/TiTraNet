class TrajectoryDatasetEval(Dataset):
    """
    Full-trajectory loader for SLIDING evaluation.
    - 한 파일 = 한 시퀀스 (전체 프레임 사용)
    - 모든 프레임에 존재하는 에이전트만 포함
    - obs_traj: [N, 2, obs_len]
    - pred_traj: [N, 2, T_full - obs_len]   # 나머지 전부
    - vx, vy, ax, ay, size, agent_type, traffic_state: [N, C, T_full]  # 전체 길이
    """
    def __init__(self, data_dir, obs_len=12, pred_len=12, min_ped=1, delim='\t'):
        super().__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.delim = delim
        self.min_ped = min_ped

        # 파일 목록
        all_files = os.listdir(self.data_dir)
        self.files = [os.path.join(self.data_dir, _path) for _path in all_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        data = read_file(path, self.delim)  # (frame, id, x, y, vx, vy, ax, ay, size_w, size_h, agent_type, traffic_state)
        frames = np.unique(data[:, 0]).tolist()
        frames.sort()

        # 프레임별 데이터 모음
        frame_data = []
        for fr in frames:
            frame_data.append(data[data[:, 0] == fr, :])

        # 모든 프레임에 존재하는 에이전트만 선택
        vehs_all = np.unique(data[:, 1]).tolist()
        keep_ids = []
        for vid in vehs_all:
            # 각 프레임에 존재하는지 확인
            present_all = True
            for fr in frames:
                if not np.any((data[:, 0] == fr) & (data[:, 1] == vid)):
                    present_all = False
                    break
            if present_all:
                keep_ids.append(vid)

        num_vehs = len(keep_ids)
        T_full = len(frames)

        if num_vehs < self.min_ped or T_full <= self.obs_len:
            # 비어있는 경우: dummy라도 만들기보단 예외를 던져 상위에서 skip 시키게 하는 게 낫습니다.
            # 여기서는 간단히 빈 텐서를 반환해도 되고, 평가 루프에서 길이 체크 후 continue 처리해도 됩니다.
            # 간단히 zeros 반환:
            return (
                torch.zeros(0, 2, self.obs_len), torch.zeros(0, 2, 0),
                torch.zeros(0, 2, self.obs_len), torch.zeros(0, 2, 0),
                torch.zeros(0), torch.zeros(0, T_full),
                torch.zeros(T_full, 0, 1), torch.zeros(T_full, 0, 1),
                torch.zeros(T_full, 0, 1), torch.zeros(T_full, 0, 1),
                torch.zeros(T_full, 0, 1), torch.zeros(T_full, 0, 2),
                torch.zeros(T_full, 0, 1)
            )

        # 텐서 준비
        curr_xy = np.zeros((num_vehs, 2, T_full), dtype=np.float32)
        curr_rel = np.zeros((num_vehs, 2, T_full), dtype=np.float32)
        curr_vx = np.zeros((num_vehs, 1, T_full), dtype=np.float32)
        curr_vy = np.zeros((num_vehs, 1, T_full), dtype=np.float32)
        curr_ax = np.zeros((num_vehs, 1, T_full), dtype=np.float32)
        curr_ay = np.zeros((num_vehs, 1, T_full), dtype=np.float32)
        curr_size = np.zeros((num_vehs, 2, T_full), dtype=np.float32)
        curr_agent_type = np.zeros((num_vehs, 1, T_full), dtype=np.float32)
        curr_traffic_state = np.zeros((num_vehs, 1, T_full), dtype=np.float32)
        curr_loss_mask = np.ones((num_vehs, T_full), dtype=np.float32)  # 전체 프레임 사용

        # 채우기
        vid2idx = {vid: i for i, vid in enumerate(keep_ids)}
        for t, fr in enumerate(frames):
            fr_rows = frame_data[t]
            for row in fr_rows:
                fid, vid = row[0], row[1]
                if vid not in vid2idx:
                    continue
                i = vid2idx[vid]
                x, y = row[2], row[3]
                vx, vy = row[4], row[5]
                ax, ay = row[6], row[7]
                w, h = row[8], row[9]
                at = row[10]
                ts = row[11]
                curr_xy[i, 0, t] = x
                curr_xy[i, 1, t] = y
                curr_vx[i, 0, t] = vx
                curr_vy[i, 0, t] = vy
                curr_ax[i, 0, t] = ax
                curr_ay[i, 0, t] = ay
                curr_size[i, 0, t] = w
                curr_size[i, 1, t] = h
                curr_agent_type[i, 0, t] = at
                curr_traffic_state[i, 0, t] = ts

        # 상대좌표
        curr_rel[:, :, 1:] = curr_xy[:, :, 1:] - curr_xy[:, :, :-1]

        # non-linear flag(평가에서 안 써도 형식 유지). 여기선 마지막 pred_len 구간 기준
        non_linear = []
        for i in range(num_vehs):
            traj = curr_xy[i]  # [2, T_full]
            use_len = min(self.pred_len, T_full)
            non_linear.append(poly_fit(traj, use_len, threshold=30.0))
        non_linear = np.asarray(non_linear, dtype=np.float32)

        # 분할: obs/pred
        obs_traj = curr_xy[:, :, :self.obs_len]                         # [N, 2, obs]
        pred_traj = curr_xy[:, :, self.obs_len:]                        # [N, 2, T_full-obs]
        obs_traj_rel = curr_rel[:, :, :self.obs_len]                    # [N, 2, obs]
        pred_traj_rel = curr_rel[:, :, self.obs_len:]                   # [N, 2, T_full-obs]

        # torch 변환
        obs_traj = torch.from_numpy(obs_traj).float()
        pred_traj = torch.from_numpy(pred_traj).float()
        obs_traj_rel = torch.from_numpy(obs_traj_rel).float()
        pred_traj_rel = torch.from_numpy(pred_traj_rel).float()
        non_linear_ped = torch.from_numpy(non_linear).float()
        loss_mask = torch.from_numpy(curr_loss_mask).float()

        vx = torch.from_numpy(curr_vx).float()
        vy = torch.from_numpy(curr_vy).float()
        ax = torch.from_numpy(curr_ax).float()
        ay = torch.from_numpy(curr_ay).float()
        size = torch.from_numpy(curr_size).float()
        agent_type = torch.from_numpy(curr_agent_type).float()
        traffic_state = torch.from_numpy(curr_traffic_state).float()

        # seq_collate와 호환되는 반환(tuple)
        return (
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
            non_linear_ped, loss_mask,
            vx, vy, ax, ay, agent_type, size, traffic_state
        )