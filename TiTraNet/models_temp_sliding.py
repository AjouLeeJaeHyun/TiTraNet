import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== UTILITIES =====================
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def _to_SB(x, name="tensor"):
    """
    허용: (S,B) 또는 (S,B,1)
    결과: (S,B)
    """
    if x.dim() == 3 and x.size(-1) == 1:
        return x.squeeze(-1)
    if x.dim() == 2:
        return x
    raise ValueError(f"{name}: expected (S,B) or (S,B,1), got {tuple(x.shape)}")

def _ensure_col(x, name="tensor"):
    """
    허용: (S,B) 또는 (S,B,1)
    결과: (S,B,1)
    """
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3 and x.size(-1) == 1:
        return x
    raise ValueError(f"{name}: expected (S,B) or (S,B,1), got {tuple(x.shape)}")


# ===================== ENCODERS =====================
class TransformerEncoder(nn.Module):
    def __init__(self, obs_len, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8,
            dim_feedforward=h_dim * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, obs_traj):
        # obs_traj: [S, B, 2]
        S, B, _ = obs_traj.shape
        emb = self.spatial_embedding(obs_traj.reshape(-1, 2))
        emb = emb.view(S, B, -1)
        return self.transformer_encoder(emb)  # (S,B,D)


class TrafficEncoder(nn.Module):
    def __init__(self, traffic_state_dim=5, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.traffic_embedding = nn.Embedding(traffic_state_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8,
            dim_feedforward=h_dim * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, traffic_state):
        # traffic_state: (S,B) or (S,B,1)
        traffic_state = _to_SB(traffic_state, name="traffic_state").long()
        S, B = traffic_state.shape
        emb = self.traffic_embedding(traffic_state.reshape(-1))
        emb = emb.view(S, B, -1)
        return self.transformer_encoder(emb)  # (S,B,D)


class VehicleEncoder(nn.Module):
    def __init__(self, agent_type_dim=6, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.agent_type_embedding = nn.Embedding(agent_type_dim, embedding_dim)
        self.size_layer = nn.Linear(2, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * embedding_dim, nhead=8,
            dim_feedforward=h_dim * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, agent_type, size):
        # agent_type: (S,B) or (S,B,1)
        # size: (S,B,2)  (필요시 보정)
        if agent_type.dim() == 3 and agent_type.size(-1) == 1:
            agent_type = agent_type.squeeze(-1)  # (S,B)
        S, B = agent_type.shape
        agent_type = agent_type.long()

        type_emb = self.agent_type_embedding(agent_type.reshape(-1)).view(S, B, -1)

        if size.dim() == 2:
            size = size.view(S, B, 2)
        size_emb = self.size_layer(size)  # (S,B,D)

        combined = torch.cat([type_emb, size_emb], dim=-1)  # (S,B,2D)
        return self.transformer_encoder(combined)


class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.state_layer = nn.Linear(4, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8,
            dim_feedforward=h_dim * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, state):
        # state: (S,B,4)  (vx,vy,ax,ay)
        return self.transformer_encoder(self.state_layer(state))


# ===================== DECODER =====================
class TransformerDecoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(embedding_dim, 2)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=8,
            dim_feedforward=mlp_dim, dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    @staticmethod
    def _causal_mask(T, device):
        return nn.Transformer.generate_square_subsequent_mask(T).to(device)

    def forward(self, last_pos, last_pos_rel, memory,
                seq_start_end, vx, vy,
                targets_rel=None, teacher_forcing_ratio=0.0):
        """
        Autoregressive decoding with optional teacher forcing
        Returns: (T,B,2)
        """
        B = last_pos.size(0)
        T = self.seq_len
        device = memory.device
        preds = []
        inputs_emb = []
        cur_rel = last_pos_rel
        for t in range(T):
            if (targets_rel is not None) and (t > 0) and (torch.rand(1, device=device) < teacher_forcing_ratio):
                cur_rel = targets_rel[t-1]
            inputs_emb.append(self.spatial_embedding(cur_rel).unsqueeze(1))  # (B,1,D)
            tgt_prefix = torch.cat(inputs_emb, dim=1)                        # (B,t+1,D)
            tgt_mask = self._causal_mask(tgt_prefix.size(1), device)         # (t+1,t+1)
            dec_out = self.transformer_decoder(tgt=tgt_prefix, memory=memory, tgt_mask=tgt_mask)
            step_rel = self.hidden2pos(dec_out[:, -1, :])                    # (B,2)
            preds.append(step_rel)
            cur_rel = step_rel
        return torch.stack(preds, dim=0)  # (T,B,2)


# ===================== ROUTING =====================
class CrossAttentionRouter(nn.Module):
    def __init__(self, query_dim, context_dim, hidden_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x, context):
        # x: (S,B,Dq)  context: (B, Dc)
        S, B, _ = x.shape
        q = self.query_proj(x)                       # (S,B,H)
        k = self.context_proj(context).unsqueeze(0)  # (1,B,H)
        k = k.expand(S, -1, -1)
        scores = (q * k).sum(-1) / self.temperature  # (S,B)
        attn = F.softmax(scores, dim=0).unsqueeze(-1)
        return x * attn


# ===================== GENERATOR =====================
class TrajectoryGenerator(nn.Module):
    def __init__(self, obs_len, pred_len,
                 embedding_dim=64, encoder_h_dim=64,
                 decoder_h_dim=128, mlp_dim=1024,
                 num_layers=1, dropout=0.0, traffic_h_dim=64):
        super().__init__()
        self.obs_len, self.pred_len = obs_len, pred_len
        self.encoder = TransformerEncoder(obs_len, embedding_dim, encoder_h_dim, num_layers, dropout)
        self.traffic_encoder = TrafficEncoder(5, embedding_dim, traffic_h_dim)
        self.vehicle_encoder = VehicleEncoder(6, embedding_dim, 64)
        self.state_encoder = StateEncoder(embedding_dim, 64)
        self.decoder = TransformerDecoder(pred_len, embedding_dim, decoder_h_dim, mlp_dim, num_layers, dropout)

        self.expected_feature_dim = 320
        self.routing = CrossAttentionRouter(self.expected_feature_dim, 128)
        self.memory_projection = nn.Linear(self.expected_feature_dim, embedding_dim)
        self.centripetal_proj = nn.Linear(1, 64)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end,
                vx, vy, ax, ay, agent_type, size, traffic_state,
                gt_pred_rel=None, teacher_forcing_ratio=0.0):
        """
        Standard single-window forward: obs_len → pred_len
        Shapes:
          obs_traj, obs_traj_rel: (S,B,2)
          vx,vy,ax,ay: (S,B,1) 또는 (S,B) → concat 시 (S,B,4)로 맞춰 사용
          agent_type: (S,B) or (S,B,1)
          size: (S,B,2)  traffic_state: (S,B) or (S,B,1)
        """
        # --- encoders ---
        enc = self.encoder(obs_traj_rel)  # (S,B,64)

        # 보조 입력 정리
        vx, vy, ax, ay = map(lambda t: _ensure_col(t, name="state"), [vx, vy, ax, ay])
        stt = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))   # (S,B,64)
        veh = self.vehicle_encoder(agent_type, size)                   # (S,B,128)
        trf = self.traffic_encoder(traffic_state)                      # (S,B,64)

        combined = torch.cat([enc, veh, stt, trf], dim=2)              # (S,B,320)

        # --- context (traffic + centripetal) ---
        vx_last, vy_last = vx[-1].squeeze(-1), vy[-1].squeeze(-1)
        ax_last, ay_last = ax[-1].squeeze(-1), ay[-1].squeeze(-1)
        g_cent = (vx_last * ay_last - vy_last * ax_last).view(-1, 1)    # (B,1)

        trf_ctx = trf.mean(0)                                          # (B,64)
        g_proj = self.centripetal_proj(g_cent)                         # (B,64)
        ctx = torch.cat([trf_ctx, g_proj], dim=1)                      # (B,128)

        combined = self.routing(combined, ctx)                         # (S,B,320)
        memory = self.memory_projection(combined).permute(1, 0, 2)     # (B,S,D)

        # --- decode ---
        return self.decoder(
            last_pos=obs_traj[-1], last_pos_rel=obs_traj_rel[-1],
            memory=memory, seq_start_end=seq_start_end,
            vx=vx, vy=vy, targets_rel=gt_pred_rel,
            teacher_forcing_ratio=teacher_forcing_ratio
        )  # (T,B,2)

    @torch.no_grad()
    def predict_sliding_windows(
        self,
        full_traj, full_traj_rel,           # (T_total,B,2)
        vx, vy, ax, ay,                     # (T_total,B) or (T_total,B,1)
        agent_type, size, traffic_state,    # (T_total,B) or (T_total,B,1), (T_total,B,2), (T_total,B) or (T_total,B,1)
        seq_start_end,
        obs_len: int = None,
        pred_len: int = None,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.0
    ):
        """
        데이터 시퀀스 전체에서 슬라이딩 윈도우 생성 후 각 윈도우마다 예측을 수행.
        반환:
          preds_rel: (N_win, pred_len, B, 2)
          win_starts: (N_win,)  각 윈도우의 시작 인덱스 (관측구간 시작 프레임)
        """
        self.eval()
        obs_len = obs_len or self.obs_len
        pred_len = pred_len or self.pred_len

        T_total, B, _ = full_traj.shape
        N_win = max(0, (T_total - (obs_len + pred_len)) // stride + 1)

        preds = []
        starts = []
        for w in range(N_win):
            s = w * stride
            e_obs = s + obs_len

            obs_traj      = full_traj[s:e_obs]          # (S,B,2)
            obs_traj_rel  = full_traj_rel[s:e_obs]      # (S,B,2)

            vx_win = vx[s:e_obs];  vy_win = vy[s:e_obs]
            ax_win = ax[s:e_obs];  ay_win = ay[s:e_obs]
            vx_win, vy_win, ax_win, ay_win = map(lambda t: _ensure_col(t, name="state_win"), [vx_win, vy_win, ax_win, ay_win])

            agent_win   = _to_SB(agent_type[s:e_obs], name="agent_type_win")   # (S,B)
            size_win    = size[s:e_obs]                                        # (S,B,2)
            traf_win    = _to_SB(traffic_state[s:e_obs], name="traffic_state_win")  # (S,B)

            pred_rel = self.forward(
                obs_traj=obs_traj, obs_traj_rel=obs_traj_rel, seq_start_end=seq_start_end,
                vx=vx_win, vy=vy_win, ax=ax_win, ay=ay_win,
                agent_type=agent_win, size=size_win, traffic_state=traf_win,
                gt_pred_rel=None, teacher_forcing_ratio=teacher_forcing_ratio
            )  # (pred_len,B,2)

            preds.append(pred_rel.unsqueeze(0))  # (1,pred_len,B,2)
            starts.append(s)

        if len(preds) == 0:
            return torch.empty(0, pred_len, B, 2, device=full_traj.device), torch.tensor([], device=full_traj.device, dtype=torch.long)

        preds_rel = torch.cat(preds, dim=0)                 # (N_win,pred_len,B,2)
        win_starts = torch.tensor(starts, device=full_traj.device, dtype=torch.long)
        return preds_rel, win_starts


# ===================== DISCRIMINATOR =====================
class TrajectoryDiscriminator(nn.Module):
    def __init__(self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
                 num_layers=1, activation='relu', batch_norm=True, dropout=0.0):
        super().__init__()
        self.seq_len = obs_len + pred_len
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8,
            dim_feedforward=mlp_dim, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, 1, embedding_dim))
        self.real_classifier = make_mlp([embedding_dim, mlp_dim, 1],
                                        activation=activation,
                                        batch_norm=batch_norm,
                                        dropout=dropout)

    def forward(self, traj_rel):
        # traj_rel: (S,B,2), S=obs+pred
        S, B, _ = traj_rel.shape
        x = self.spatial_embedding(traj_rel).view(S, B, -1)
        x = x + self.positional_encoding[:S]
        h = self.encoder(x)
        return self.real_classifier(h[-1])  # (B,1)