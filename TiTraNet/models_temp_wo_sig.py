import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ CUDA 에러를 동기화해서 실제 난 줄을 정확히 표시
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


# ===================== ENCODERS =====================
class TransformerEncoder(nn.Module):
    def __init__(self, obs_len, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.obs_len = obs_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=h_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        # (선택) batch_first=True로 바꾸고 싶은 경우, 위 레이어들 정의에서 batch_first=True 지정 필요

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: (obs_len, batch, 2)
        Output:
        - final_h: (obs_len, batch, embedding_dim)
        """
        batch_size = obs_traj.shape[1]
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(self.obs_len, batch_size, self.embedding_dim)
        final_h = self.transformer_encoder(obs_traj_embedding)
        return final_h


class TrafficEncoder(nn.Module):
    def __init__(self, traffic_state_dim=5, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(TrafficEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.traffic_embedding = nn.Embedding(traffic_state_dim, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=h_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, traffic_state):
        traffic_state = traffic_state.long()
        batch_size = traffic_state.size(1)
        seq_len = traffic_state.size(0)

        flat_traffic_state = traffic_state.reshape(-1)
        traffic_state_embedding = self.traffic_embedding(flat_traffic_state)
        traffic_state_embedding = traffic_state_embedding.view(seq_len, batch_size, -1)
        final_h = self.transformer_encoder(traffic_state_embedding)
        return final_h


class VehicleEncoder(nn.Module):
    def __init__(self, agent_type_dim=8, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(VehicleEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.agent_type_dim = agent_type_dim  # ✅ forward에서 범위 가드에 사용

        self.agent_type_embedding = nn.Embedding(agent_type_dim, embedding_dim)
        self.size_layer = nn.Linear(2, embedding_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * embedding_dim,
            nhead=8,
            dim_feedforward=h_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, agent_type, size):
        # ✅ 임베딩 인덱스 안전 가드 (응급처치). 근본 해결은 데이터 정제/차원 확장
        agent_type = agent_type.long().clamp_(0, self.agent_type_dim - 1)

        batch_size = agent_type.size(1)

        # agent_type: (seq_len, batch) 혹은 (seq_len, batch, 1)
        if agent_type.dim() == 3 and agent_type.size(-1) == 1:
            agent_type = agent_type.squeeze(-1)

        agent_type_embedding = self.agent_type_embedding(agent_type)  # (seq_len, batch, emb)
        size_embedding = self.size_layer(size)                        # (seq_len, batch, emb)

        combined_embedding = torch.cat([agent_type_embedding, size_embedding], dim=-1)  # (seq_len, batch, 2*emb)
        final_h = self.transformer_encoder(combined_embedding)
        return final_h


class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(StateEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.state_layer = nn.Linear(4, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=h_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, state):
        """
        Inputs:
        - state: (seq_len, batch, 4)  # (vx, vy, ax, ay)
        Output:
        - final_h: (seq_len, batch, embedding_dim)
        """
        state_embedding = self.state_layer(state)
        final_h = self.transformer_encoder(state_embedding)
        return final_h


# ===================== DECODER =====================
class TransformerDecoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(TransformerDecoder, self).__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(embedding_dim, 2)

    def forward(self, last_pos, last_pos_rel, memory, seq_start_end, vx, vy):
        """
        Inputs:
        - last_pos: (batch, 2)
        - last_pos_rel: (batch, 2)
        - memory: (obs_len, batch, embedding_dim)
        """
        batch = last_pos.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel)      # (batch, emb)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)  # (1, batch, emb)

        device = last_pos_rel.device
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len).to(device)

        decoder_output = self.transformer_decoder(
            decoder_input.expand(self.seq_len, -1, -1),  # (pred_len, batch, emb)
            memory,                                      # (obs_len,  batch, emb)
            tgt_mask=tgt_mask
        )
        pred_traj_fake_rel = self.hidden2pos(decoder_output)      # (pred_len, batch, 2)
        return pred_traj_fake_rel


# ===================== ROUTING =====================
class CrossAttentionRouter(nn.Module):
    """
    입력 차원을 런타임에 자동 추론하는 크로스어텐션 라우터.
    - query_proj:  (Q -> H)  with LazyLinear
    - context_proj:(C -> H)  with LazyLinear
    출력은 온도-스케일 가중합으로 원래 x의 차원(Q)을 그대로 유지.
    """
    def __init__(self, hidden_dim=128):
        super(CrossAttentionRouter, self).__init__()
        self.query_proj   = nn.LazyLinear(hidden_dim)
        self.context_proj = nn.LazyLinear(hidden_dim)
        self.temperature  = nn.Parameter(torch.ones(1) * 1.0)  # learnable scalar

    def forward(self, x, context):
        """
        x:       (seq_len, batch, Q)  # Q는 256/320 등 가변
        context: (batch, C)           # C는 1/64/65 등 가변
        return:  (seq_len, batch, Q)
        """
        seq_len, batch_size, _ = x.size()

        # LazyLinear는 2D 입력을 기대하므로 (S*B, Q)로 펼쳤다가 복원
        q = self.query_proj(x.reshape(-1, x.size(-1))).view(seq_len, batch_size, -1)  # (S,B,H)
        k = self.context_proj(context).unsqueeze(0).expand(seq_len, -1, -1)           # (S,B,H)

        scores = torch.einsum('sbd,sbd->sb', q, k) / self.temperature                 # (S,B)
        attn_weights = F.softmax(scores, dim=0).unsqueeze(2)                          # (S,B,1)

        return x * attn_weights                                                       # (S,B,Q)


# ===================== GENERATOR =====================
class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, dropout=0.0,
        traffic_h_dim=64, use_traffic_signal=False
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.traffic_h_dim = traffic_h_dim
        self.use_traffic_signal = use_traffic_signal

        # Encoders
        self.encoder         = TransformerEncoder(obs_len, embedding_dim, encoder_h_dim, num_layers, dropout)   # -> (obs_len, B, 64)
        self.traffic_encoder = TrafficEncoder(traffic_state_dim=5, embedding_dim=embedding_dim, h_dim=traffic_h_dim)
        self.vehicle_encoder = VehicleEncoder(8, embedding_dim, 64)   # -> (obs_len, B, 2*emb=128)
        self.state_encoder   = StateEncoder(embedding_dim, 64)        # -> (obs_len, B, 64)

        # Decoder
        self.decoder = TransformerDecoder(pred_len, embedding_dim, decoder_h_dim, mlp_dim, num_layers, dropout)

        # ✅ combined_dim은 상황(traffic on/off)에 따라 256/320 등 가변일 수 있으므로,
        #    아래 라우터/프로젝션은 LazyLinear를 쓰는 모듈에 맡긴다.
        self.routing = CrossAttentionRouter(hidden_dim=128)
        self.memory_projection = nn.LazyLinear(self.embedding_dim)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state):
        batch = obs_traj_rel.size(1)
        device = obs_traj_rel.device
        dtype  = obs_traj_rel.dtype

        # ✅ 입력 값 범위/유효성 검증 (GPU assert 전에 명시적 에러로 막기)
        with torch.no_grad():
            at_min = int(agent_type.min().item())
            at_max = int(agent_type.max().item())
            if at_min < 0 or at_max >= 8:  # VehicleEncoder(agent_type_dim=8)
                raise ValueError(
                    f"[agent_type out of range] min={at_min}, max={at_max}, expected in [0, 7]. "
                    f"데이터를 정제하거나 VehicleEncoder(agent_type_dim)를 max+1로 확장하세요."
                )
            if self.use_traffic_signal:
                ts_min = int(traffic_state.min().item())
                ts_max = int(traffic_state.max().item())
                if ts_min < 0 or ts_max >= 5:  # TrafficEncoder(traffic_state_dim=5)
                    raise ValueError(
                        f"[traffic_state out of range] min={ts_min}, max={ts_max}, expected in [0, 4]."
                    )

        # Encodings
        final_encoder_h  = self.encoder(obs_traj_rel)                             # (obs_len, B, 64)
        vehicle_encoding = self.vehicle_encoder(agent_type, size)                 # (obs_len, B, 128)

        # ✅ state 텐서 dtype/device 정합화
        vx = vx.to(device=device, dtype=dtype)
        vy = vy.to(device=device, dtype=dtype)
        ax = ax.to(device=device, dtype=dtype)
        ay = ay.to(device=device, dtype=dtype)

        state_encoding   = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2)) # (obs_len, B, 64)

        if self.use_traffic_signal:
            traffic_encoding = self.traffic_encoder(traffic_state)                # (obs_len, B, 64)
        else:
            traffic_encoding = torch.zeros(self.obs_len, batch, self.embedding_dim, device=device, dtype=dtype)

        # 모두 (seq_len, B, C) 형태 보장
        if traffic_encoding.dim() == 2:
            traffic_encoding = traffic_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)
        if vehicle_encoding.dim() == 2:
            vehicle_encoding = vehicle_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)
        if state_encoding.dim() == 2:
            state_encoding = state_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)

        # Concat → combined_encoding: (obs_len, B, ?(=256/320))
        combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding, traffic_encoding], dim=2)

        # Context = traffic_context(mean over time) + centripetal scalar
        if self.use_traffic_signal:
            traffic_context = traffic_encoding.mean(dim=0)                        # (B, 64)
        else:
            traffic_context = torch.zeros(batch, self.embedding_dim, device=device, dtype=dtype)

        # centripetal = v x a (2D 외적 스칼라)
        vx_last = vx[-1]
        vy_last = vy[-1]
        ax_last = ax[-1]
        ay_last = ay[-1]
        g_centripetal = vx_last * ay_last - vy_last * ax_last                     # (B,)
        g_centripetal = g_centripetal.reshape(batch, 1).to(device=device, dtype=dtype)

        context = torch.cat([traffic_context, g_centripetal], dim=1)              # (B, 65)

        # Routing & projection (입력 차원 자동 적응)
        combined_routed = self.routing(combined_encoding, context)                # (obs_len, B, ?)
        memory_for_decoder = self.memory_projection(
            combined_routed.reshape(-1, combined_routed.size(-1))
        ).view(self.obs_len, batch, self.embedding_dim)                           # (obs_len, B, 64)

        # Decoder
        pred_traj_fake_rel = self.decoder(
            last_pos=obs_traj[-1],
            last_pos_rel=obs_traj_rel[-1],
            memory=memory_for_decoder,
            seq_start_end=seq_start_end,
            vx=vx, vy=vy
        )
        return pred_traj_fake_rel


# ===================== DISCRIMINATOR =====================
class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.embedding_dim = embedding_dim

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, 1, embedding_dim))
        real_classifier_dims = [embedding_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj_rel: (obs_len + pred_len, batch, 2)
        Output:
        - scores: (batch, 1)
        """
        batch_size = traj_rel.shape[1]

        if traj_rel.shape[-1] != self.embedding_dim:
            traj_rel = self.spatial_embedding(traj_rel)

        traj_rel_embedded = traj_rel.reshape(self.seq_len, batch_size, self.embedding_dim)
        traj_rel_embedded = traj_rel_embedded + self.positional_encoding

        final_h = self.encoder(traj_rel_embedded)     # (seq_len, batch, emb)
        classifier_input = final_h[-1, :, :]          # (batch, emb)
        scores = self.real_classifier(classifier_input)  # (batch, 1)
        return scores