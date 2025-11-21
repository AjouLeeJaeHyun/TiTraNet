import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUDA_LAUNCH_BLOCKING=1



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
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

# 이게 새로 짠 encoder
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

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (obs_len, batch, h_dim)
        """
        batch_size = obs_traj.shape[1]

        # 2D 좌표를 embedding_dim으로 변환
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(self.obs_len, batch_size, self.embedding_dim)

        # Transformer Encoder 실행
        final_h = self.transformer_encoder(obs_traj_embedding)  # (obs_len, batch, embedding_dim)

        return final_h

class TrafficEncoder(nn.Module):
    def __init__(self, traffic_state_dim=5, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(TrafficEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 트래픽 상태를 위한 임베딩 레이어
        self.traffic_embedding = nn.Embedding(traffic_state_dim, embedding_dim)

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,  # 임베딩 차원과 동일하게 설정
            nhead=8,  # Multi-Head Attention 헤드 수
            dim_feedforward=h_dim * 4,  # 피드포워드 네트워크 차원
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, traffic_state):
        traffic_state = traffic_state.long()

        batch_size = traffic_state.size(1)
        seq_len = traffic_state.size(0)

        # 임베딩 변환
        flat_traffic_state = traffic_state.reshape(-1)
        traffic_state_embedding = self.traffic_embedding(flat_traffic_state)

        # Transformer 입력 형태로 변환 (seq_len, batch, embedding_dim)
        traffic_state_embedding = traffic_state_embedding.view(seq_len, batch_size, -1)

        # Transformer Encoder 실행
        final_h = self.transformer_encoder(traffic_state_embedding)  # (seq_len, batch, embedding_dim)

        return final_h

class VehicleEncoder(nn.Module):
    def __init__(self, agent_type_dim=6, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(VehicleEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 차량 유형 및 크기 임베딩 레이어
        self.agent_type_embedding = nn.Embedding(agent_type_dim, embedding_dim)
        self.size_layer = nn.Linear(2, embedding_dim)  # 차량 크기 인코딩

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * embedding_dim,  # 차량 타입과 크기 임베딩을 합친 차원
            nhead=8,  # Multi-Head Attention 헤드 수
            dim_feedforward=h_dim * 4,  # FFN 내부 차원
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, agent_type, size):
        agent_type = agent_type.long()
        batch_size = agent_type.size(1)
        seq_len = agent_type.size(0)

        # 차량 타입을 임베딩 벡터로 변환
        agent_type_embedding = self.agent_type_embedding(agent_type)
        agent_type_embedding = torch.squeeze(agent_type_embedding, -2)  # (seq_len, batch, embedding_dim)

        # 차량 크기 정보를 임베딩 차원으로 변환
        size_embedding = self.size_layer(size)  # (seq_len, batch, embedding_dim)

        # 두 정보를 결합하여 Transformer 입력 형식으로 맞춤
        combined_embedding = torch.cat([agent_type_embedding, size_embedding], dim=-1)  # (seq_len, batch, 2 * embedding_dim)

        # Transformer Encoder 실행
        final_h = self.transformer_encoder(combined_embedding)  # (seq_len, batch, 2 * embedding_dim)

        return final_h


class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(StateEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 속도(vx, vy)와 가속도(ax, ay)를 Transformer 입력 차원으로 변환
        self.state_layer = nn.Linear(4, embedding_dim)  # (vx, vy, ax, ay → embedding_dim)

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,  # 입력 차원
            nhead=8,  # Multi-Head Attention 헤드 수
            dim_feedforward=h_dim * 4,  # FFN 내부 차원
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, state):
        """
        Inputs:
        - state: Tensor of shape (seq_len, batch, 4)  # (vx, vy, ax, ay)

        Output:
        - final_h: Tensor of shape (seq_len, batch, embedding_dim)
        """
        batch = state.size(1)
        seq_len = state.size(0)

        # (vx, vy, ax, ay) → embedding_dim 차원으로 변환
        state_embedding = self.state_layer(state)  # (seq_len, batch, embedding_dim)

        # Transformer Encoder 실행
        final_h = self.transformer_encoder(state_embedding)  # (seq_len, batch, embedding_dim)

        return final_h

#새로 추가한 Transformer를 사용하는 Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(TransformerDecoder, self).__init__()

        self.seq_len = seq_len  # 예측할 길이 (ex. 12)
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim

        # Transformer Decoder 사용
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,  # 모델 차원 (LSTM의 hidden_dim 대체)
            nhead=8,  # Multi-Head Attention의 헤드 수
            dim_feedforward=mlp_dim,  # FFN 내부 차원
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # 입력 임베딩
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        # 최종 출력 변환 (hidden_dim → 2D 좌표)
        self.hidden2pos = nn.Linear(embedding_dim, 2)

    def forward(self, last_pos, last_pos_rel, memory, seq_start_end, vx, vy):
        """
        Inputs:
        - last_pos: Tensor (batch, 2) → 이전 위치
        - last_pos_rel: Tensor (batch, 2) → 이전 상대 위치
        - memory: Transformer Encoder에서 추출된 Feature (obs_len, batch, embedding_dim)
        - seq_start_end: 시퀀스 인덱스 정보
        - vx, vy: 속도 정보 (optional)

        Output:
        - pred_traj_fake_rel: (pred_len, batch, 2) → 예측된 상대 경로
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []

        # 이전 위치를 Transformer 입력 차원으로 변환
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        # Transformer는 미래 정보가 보이지 않도록 Mask 필요
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len).cuda()

        # Transformer Decoder 실행
        decoder_output = self.transformer_decoder(
            decoder_input.expand(self.seq_len, -1, -1),  # (pred_len, batch, embedding_dim)
            memory,  # Transformer Encoder Output (obs_len, batch, embedding_dim)
            tgt_mask=tgt_mask
        )

        # Transformer 출력 → 2D 좌표 변환
        pred_traj_fake_rel = self.hidden2pos(decoder_output)

        return pred_traj_fake_rel

# class Decoder(nn.Module):
#     """Decoder is part of TrajectoryGenerator"""
#     def __init__(
#         self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
#         pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
#         activation='relu', batch_norm=True, pooling_type='atten_net',
#         neighborhood_size=2.0, grid_size=8
#     ):
#         super(Decoder, self).__init__()

#         self.seq_len = seq_len
#         self.mlp_dim = mlp_dim
#         self.h_dim = h_dim
#         self.embedding_dim = embedding_dim
#         self.pool_every_timestep = pool_every_timestep

#         self.decoder = nn.LSTM(
#             embedding_dim, h_dim, num_layers, dropout=dropout
#         )

#         if pool_every_timestep:
#             if pooling_type == 'atten_net':
#                 self.pool_net = AttenPoolNet(
#                     embedding_dim=self.embedding_dim,
#                     h_dim=self.h_dim,
#                     mlp_dim=mlp_dim,
#                     bottleneck_dim=bottleneck_dim,
#                     activation=activation,
#                     batch_norm=batch_norm,
#                     dropout=dropout
#                 )


#             mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
#             self.mlp = make_mlp(
#                 mlp_dims,
#                 activation=activation,
#                 batch_norm=batch_norm,
#                 dropout=dropout
#             )

#         self.spatial_embedding = nn.Linear(2, embedding_dim)
#         self.hidden2pos = nn.Linear(h_dim, 2)

#     def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, vx,vy):
#         """
#         Inputs:
#         - last_pos: Tensor of shape (batch, 2) #前一个位置
#         - last_pos_rel: Tensor of shape (batch, 2) #相对位置
#         - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim) #隐藏状态和单元状态
#         - seq_start_end: A list of tuples which delimit sequences within batch #序列开始和节俗的索引
#         Output:
#         - pred_traj: tensor of shape (self.seq_len, batch, 2)
#         """
#         batch = last_pos.size(0)
#         pred_traj_fake_rel = []
#         decoder_input = self.spatial_embedding(last_pos_rel)
#         decoder_input = decoder_input.view(1, batch, self.embedding_dim)

#         for _ in range(self.seq_len):
#             output, state_tuple = self.decoder(decoder_input, state_tuple)

#             rel_pos = self.hidden2pos(output.view(-1, self.h_dim))

#             curr_pos = rel_pos + last_pos


#             if self.pool_every_timestep:
#                 decoder_h = state_tuple[0]
#                 pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos, vx, vy)

#                 decoder_h = torch.cat(
#                     [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
#                 decoder_h = self.mlp(decoder_h)
#                 decoder_h = torch.unsqueeze(decoder_h, 0)
#                 state_tuple = (decoder_h, state_tuple[1])


#             embedding_input = rel_pos


#             decoder_input = self.spatial_embedding(embedding_input)
#             decoder_input = decoder_input.view(1, batch, self.embedding_dim)

#             pred_traj_fake_rel.append(rel_pos.view(batch, -1))
#             last_pos = curr_pos

#         pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
#         return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """

        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h



class AttenPoolNet(PoolHiddenNet):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(AttenPoolNet, self).__init__(embedding_dim, h_dim, mlp_dim, bottleneck_dim,
                                           activation, batch_norm, dropout)

        # Additional layers for processing velocity and computing attention weights
        self.velocity_embedding = nn.Linear(2, embedding_dim)
        self.attention_mlp = make_mlp(
            [embedding_dim * 2, mlp_dim, 1],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def compute_attention_weights(self, rel_pos_embedding, velocity_embedding):
        concatenated = torch.cat([rel_pos_embedding, velocity_embedding], dim=1)
        attention_scores = self.attention_mlp(concatenated)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights

    def forward(self, h_states, seq_start_end, end_pos, vx, vy):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]


            curr_hidden_repeated = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_repeated = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_transposed = curr_end_pos.repeat(1, num_ped).view(num_ped * num_ped, -1)
            curr_rel_pos = curr_end_pos_repeated - curr_end_pos_transposed
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)


            curr_vx = vx[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_vy = vy[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_velocity = torch.cat((curr_vx, curr_vy), dim=1)
            curr_velocity_embedding = self.velocity_embedding(curr_velocity)

            attention_weights = self.compute_attention_weights(curr_rel_embedding, curr_velocity_embedding)


            weighted_h_input = torch.cat([curr_rel_embedding, curr_hidden_repeated], dim=1)
            weighted_h_input *= 0.05 * attention_weights.view(-1, 1)

            # MLP processing as in PoolHiddenNet
            curr_pool_h = self.mlp_pre_pool(weighted_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]

            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


# Define the make_mlp function if not defined
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

# 여기는 새로 transformer decoder를 사용하는 trajectory Generator이다
class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, dropout=0.0,
        traffic_h_dim=64
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.traffic_h_dim = traffic_h_dim

        # ✅ 기존 LSTM Encoder를 TransformerEncoder로 변경
        self.encoder = TransformerEncoder(
            obs_len=obs_len,  # 추가
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # ✅ 기존 Traffic, Vehicle, State Encoder 추가
        self.traffic_encoder = TrafficEncoder(traffic_state_dim=5, embedding_dim=64, h_dim=traffic_h_dim)
        self.vehicle_encoder = VehicleEncoder(agent_type_dim=6, embedding_dim=64, h_dim=64)
        self.state_encoder = StateEncoder(embedding_dim=embedding_dim, h_dim=64)

        # ✅ Transformer 기반 Decoder 추가
        self.decoder = TransformerDecoder(
            seq_len=pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )


    # 안쓰는 Forard 함수 아래에 새로운 forward 함수 있음
    # def forward(self, obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state):
    #     batch = obs_traj_rel.size(1)

    #     # ✅ Transformer 기반 Encoder → Feature 추출
    #     final_encoder_h = self.encoder(obs_traj_rel)  # (obs_len=12, batch, embedding_dim)

    #     # ✅ 추가적인 Feature Encoding
    #     vehicle_encoding = self.vehicle_encoder(agent_type, size)  # (1, batch, h_dim)
    #     state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))  # (1, batch, h_dim)
    #     traffic_encoding = self.traffic_encoder(traffic_state)  # (1, batch, h_dim)

    #     # ✅ Feature 차원을 obs_len(12)과 맞추기
    #     vehicle_encoding = vehicle_encoding.expand(self.obs_len, -1, -1)  # (12, batch, h_dim)
    #     state_encoding = state_encoding.expand(self.obs_len, -1, -1)  # (12, batch, h_dim)
    #     traffic_encoding = traffic_encoding.expand(self.obs_len, -1, -1)  # (12, batch, h_dim)

    #     # ✅ Transformer에서는 Pooling 없이 Attention으로 모든 정보 학습
    #     combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding, traffic_encoding], dim=2)

    #     # ✅ `batch` 크기를 직접 가져오기
    #     batch = combined_encoding.shape[1]  

    #     # ✅ view() 적용 전 차원 확인 후 맞추기
    #     try:
    #         mlp_decoder_context_input = combined_encoding.view(self.obs_len, batch, self.encoder_h_dim * 3 + self.traffic_h_dim)
    #     except RuntimeError as e:
    #         print(f"⚠️ RuntimeError: {e}")
    #         print(f"combined_encoding.shape: {combined_encoding.shape}, expected ({self.obs_len}, {batch}, {self.encoder_h_dim * 3 + self.traffic_h_dim})")
    #         raise e

    #     # ✅ Transformer Decoder로 예측 수행
    #     pred_traj_fake_rel = self.decoder(
    #         last_pos=obs_traj[-1],
    #         last_pos_rel=obs_traj_rel[-1],
    #         memory=mlp_decoder_context_input,
    #         seq_start_end=seq_start_end,
    #         vx=vx, vy=vy
    #     )

    #     return pred_traj_fake_rel


# 아래는 기존 Trajectory Generator이다
# class TrajectoryGenerator(nn.Module):
#     def __init__(
#         self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
#         decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
#         noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
#         pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
#         activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
#         traffic_h_dim = 64
#     ):
#         super(TrajectoryGenerator, self).__init__()

#         if pooling_type and pooling_type.lower() == 'none':
#             pooling_type = None
#         self.traffic_h_dim=traffic_h_dim

#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.mlp_dim = mlp_dim
#         self.encoder_h_dim = encoder_h_dim
#         self.decoder_h_dim = decoder_h_dim
#         self.embedding_dim = embedding_dim
#         self.noise_dim = noise_dim
#         self.num_layers = num_layers
#         self.noise_type = noise_type
#         self.noise_mix_type = noise_mix_type
#         self.pooling_type = pooling_type
#         self.noise_first_dim = 0
#         self.pool_every_timestep = pool_every_timestep
#         self.bottleneck_dim = 1024

#         self.encoder = Encoder(
#             embedding_dim=embedding_dim,
#             h_dim=64,
#             mlp_dim=mlp_dim,
#             num_layers=num_layers,
#             dropout=dropout
#         )
#         self.traffic_encoder = TrafficEncoder(traffic_state_dim=5,embedding_dim=64, h_dim=traffic_h_dim)
#         self.vehicle_encoder = VehicleEncoder(agent_type_dim=6, embedding_dim=64, h_dim=64)
#         self.state_encoder = StateEncoder(embedding_dim=embedding_dim, h_dim=64)

#         self.decoder = Decoder(
#             pred_len,
#             embedding_dim=embedding_dim,
#             h_dim=decoder_h_dim,
#             mlp_dim=mlp_dim,
#             num_layers=num_layers,
#             pool_every_timestep=pool_every_timestep,
#             dropout=dropout,
#             bottleneck_dim=bottleneck_dim,
#             activation=activation,
#             batch_norm=batch_norm,
#             pooling_type=pooling_type,
#             grid_size=grid_size,
#             neighborhood_size=neighborhood_size
#         )

#         if pooling_type == 'atten_net':
#             self.pool_net = AttenPoolNet(
#                 embedding_dim=self.embedding_dim,
#                 h_dim=encoder_h_dim,
#                 mlp_dim=mlp_dim,
#                 activation=activation,
#                 bottleneck_dim=bottleneck_dim,
#                 batch_norm=batch_norm,
#                 dropout=dropout
#             )

#         if self.noise_dim[0] == 0:
#             self.noise_dim = None
#         else:
#             self.noise_first_dim = noise_dim[0]

#         # Decoder Hidden
#         if pooling_type:
#             input_dim = encoder_h_dim*3 + bottleneck_dim + traffic_h_dim

#         else:
#             input_dim = encoder_h_dim

#         if self.mlp_decoder_needed():
#             mlp_decoder_context_dims = [
#                 input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
#             ]

#             self.mlp_decoder_context = make_mlp(
#                 mlp_decoder_context_dims,
#                 activation=activation,
#                 batch_norm=batch_norm,
#                 dropout=dropout
#             ) 

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

# 새로 만든 def forward(pooling type 제거됨)
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state, user_noise=None):
        batch = obs_traj_rel.size(1)

        # ✅ 기존 LSTM Encoder → Feature 추출
        final_encoder_h = self.encoder(obs_traj_rel)

        # ✅ 추가적인 Feature Encoding
        vehicle_encoding = self.vehicle_encoder(agent_type, size)
        state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))
        traffic_encoding = self.traffic_encoder(traffic_state)

        # ✅ 디버깅용 출력
        #print(f"final_encoder_h.device: {final_encoder_h.device}")
        #print(f"vehicle_encoding.device: {vehicle_encoding.device}")
        #print(f"state_encoding.device: {state_encoding.device}")
        #print(f"traffic_encoding.device: {traffic_encoding.device}")
        
        # ✅ Feature 차원이 (obs_len, batch, feature_dim) 형태인지 확인하고 맞추기
        if len(vehicle_encoding.shape) == 2:
            vehicle_encoding = vehicle_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)
        if len(state_encoding.shape) == 2:
            state_encoding = state_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)
        if len(traffic_encoding.shape) == 2:
            traffic_encoding = traffic_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)

        # ✅ Transformer에서는 Pooling 없이 Attention으로 모든 정보 학습
        combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding, traffic_encoding], dim=2)
        #print(f"🚀 batch (from shape[1]): {combined_encoding.shape[1]}")
        # ✅ batch 크기 계산 후 view() 수행
        combined_encoding = combined_encoding[:batch * self.obs_len]  # batch 크기에 맞게 조정
        #print(f"combined_encoding.device: {combined_encoding.device}")

        # 🚀 View() 수행 전, expected feature dimension을 정확히 계산하기
        expected_feature_dim = (
            final_encoder_h.shape[2] + 
            vehicle_encoding.shape[2] + 
            state_encoding.shape[2] + 
            traffic_encoding.shape[2]
        )

        # 🔍 디버깅을 위한 출력
        #print(f"🔍 Expected feature dim: {expected_feature_dim}")  
        #print(f"✅ Combined encoding shape before view: {combined_encoding.shape}")

        # 🚀 올바른 view() 적용
        mlp_decoder_context_input = combined_encoding.view(self.obs_len, batch, expected_feature_dim)
        #print(f"mlp_decoder_context_input.device: {mlp_decoder_context_input.device}")


        # 🚀 올바른 Transformer 입력 차원으로 맞추기
        expected_embedding_dim = self.embedding_dim  # Transformer의 d_model과 맞춰야 함

        # MLP를 사용하여 memory 차원을 변환 (320 → 64)
        self.memory_projection = nn.Linear(expected_feature_dim, expected_embedding_dim).to(device)  # (320 → 64)
        #print(f"self.memory_projection.weight.device: {self.memory_projection.weight.device}")

        # Transformer Decoder 입력 크기 변환
        mlp_decoder_context_input = mlp_decoder_context_input.to(device)
        mlp_decoder_context_input = self.memory_projection(mlp_decoder_context_input)  # (obs_len, batch, 64)

        # ✅ 디버깅용 출력
        #print(f"✅ Updated mlp_decoder_context_input.shape: {mlp_decoder_context_input.shape}")

        # ✅ 디버깅용 출력
        #print(f"combined_encoding.shape: {combined_encoding.shape}")
        #print(f"Calculated batch size: {batch}")

        # ✅ Transformer Decoder로 예측 수행
        pred_traj_fake_rel = self.decoder(
            last_pos=obs_traj[-1],
            last_pos_rel=obs_traj_rel[-1],
            memory=mlp_decoder_context_input,
            seq_start_end=seq_start_end,
            vx=vx, vy=vy
        )

        return pred_traj_fake_rel

# 기존 def forward( pooling type이 추가되어 있음)
    # def forward(self, obs_traj, obs_traj_rel, seq_start_end,  vx, vy, ax, ay, agent_type, size, traffic_state, user_noise=None,):
    #     """
    #     Inputs:
    #     - obs_traj: Tensor of shape (obs_len, batch, 2)
    #     - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
    #     - seq_start_end: A list of tuples which delimit sequences within batch.
    #     - user_noise: Generally used for inference when you want to see
    #     relation between different types of noise and outputs.
    #     Output:
    #     - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
    #     """
    #     batch = obs_traj_rel.size(1)
    #     # Encode seq
    #     final_encoder_h = self.encoder(obs_traj_rel)
    #     # 使用新编码器
    #     vehicle_encoding = self.vehicle_encoder(agent_type, size)
    #     state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))
    #     traffic_encoding = self.traffic_encoder(traffic_state)


    #     combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding], dim=2)




    #     # Pool States

    #     if self.pooling_type:
    #         end_pos = obs_traj[-1, :, :]

    #         pool_h = self.pool_net(combined_encoding, seq_start_end, end_pos,vx,vy)



    #         mlp_decoder_context_input = torch.cat(
    #             [combined_encoding.view(-1, self.encoder_h_dim*3), pool_h, traffic_encoding.view(-1, self.traffic_h_dim)], dim=1)



    #     else:
    #         mlp_decoder_context_input = combined_encoding.view(
    #             -1, self.encoder_h_dim)


    #     if self.mlp_decoder_needed():
    #         noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
    #     else:
    #         noise_input = mlp_decoder_context_input

    #     decoder_h = self.add_noise(
    #         noise_input, seq_start_end, user_noise=user_noise)
    #     decoder_h = torch.unsqueeze(decoder_h, 0)

    #     decoder_c = torch.zeros(
    #         self.num_layers, batch, self.decoder_h_dim
    #     ).cuda()

    #     state_tuple = (decoder_h, decoder_c)
    #     last_pos = obs_traj[-1]
    #     last_pos_rel = obs_traj_rel[-1]
    #     # Predict Trajectory

    #     decoder_out = self.decoder(
    #         last_pos,
    #         last_pos_rel,
    #         state_tuple,
    #         seq_start_end,
    #         vx,vy
    #     )
    #     pred_traj_fake_rel, final_decoder_h = decoder_out

    #     return pred_traj_fake_rel

# 이거는 새로 짠 discriminator(transformer 사용한 것)
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

        # 기존 LSTM을 Transformer Encoder로 변경
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,  # 모델 차원 설정 (LSTM hidden_dim 대신 embedding_dim 사용)
            nhead=8,  # Multi-head Attention 개수
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Positional Encoding 추가
        self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, 1, embedding_dim))  

        # 분류기 정의 (LSTM의 최종 hidden state를 대체)
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
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        batch_size = traj_rel.shape[1]

        # 2D 위치 데이터를 Transformer 입력 차원으로 변환
        # 디버깅용 출력 추가
        #print(f"🔍 traj_rel.shape: {traj_rel.shape}")
        #print(f"🔍 Expected total elements: {self.seq_len * batch_size * self.embedding_dim}") 
        #print(f"🔍 Actual total elements: {traj_rel.numel()}") 

        # ✅ 만약 traj_rel이 (24, 740, 2)라면, 2D → 64D 변환
        if traj_rel.shape[-1] != self.embedding_dim:
        #    print(f"🚀 Applying spatial embedding to match embedding_dim={self.embedding_dim}")
            traj_rel = self.spatial_embedding(traj_rel)

        # ✅ reshape() 사용 (shape이 맞으면 자동으로 변환됨)
        traj_rel_embedded = traj_rel.reshape(self.seq_len, batch_size, self.embedding_dim)

        # Positional Encoding 추가
        traj_rel_embedded = traj_rel_embedded + self.positional_encoding

        # Transformer Encoder 적용
        final_h = self.encoder(traj_rel_embedded)  # (seq_len, batch, embedding_dim)

        # 마지막 타임스텝의 출력을 사용 (기존 LSTM의 hidden state 역할)
        classifier_input = final_h[-1, :, :]  # (batch, embedding_dim)

        # Classification
        scores = self.real_classifier(classifier_input)  # (batch, 1)

        return scores

# 기존 trajectorydiscriminator(이거는 lstm 사용한 것)
# class TrajectoryDiscriminator(nn.Module):
#     def __init__(
#         self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
#         num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
#         d_type='local'
#     ):
#         super(TrajectoryDiscriminator, self).__init__()

#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.seq_len = obs_len + pred_len
#         self.mlp_dim = mlp_dim
#         self.h_dim = h_dim
#         self.d_type = d_type

#         self.encoder = Encoder(
#             embedding_dim=embedding_dim,
#             h_dim=h_dim,
#             mlp_dim=mlp_dim,
#             num_layers=num_layers,
#             dropout=dropout
#         )

#         real_classifier_dims = [h_dim, mlp_dim, 1]
#         self.real_classifier = make_mlp(
#             real_classifier_dims,
#             activation=activation,
#             batch_norm=batch_norm,
#             dropout=dropout
#         )
#         if d_type == 'global':
#             mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
#             self.pool_net = PoolHiddenNet(
#                 embedding_dim=embedding_dim,
#                 h_dim=h_dim,
#                 mlp_dim=mlp_pool_dims,
#                 bottleneck_dim=h_dim,
#                 activation=activation,
#                 batch_norm=batch_norm
#             )

#     def forward(self, traj, traj_rel, seq_start_end=None):
#         """
#         Inputs:
#         - traj: Tensor of shape (obs_len + pred_len, batch, 2)
#         - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
#         - seq_start_end: A list of tuples which delimit sequences within batch
#         Output:
#         - scores: Tensor of shape (batch,) with real/fake scores
#         """
#         final_h = self.encoder(traj_rel)

#         if self.d_type == 'local':
#             classifier_input = final_h.squeeze()
#         else:
#             classifier_input = self.pool_net(
#                 final_h.squeeze(), seq_start_end, traj[0]
#             )
#         scores = self.real_classifier(classifier_input)
#         return scores
