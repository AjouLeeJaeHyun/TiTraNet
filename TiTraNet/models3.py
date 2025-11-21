import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

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

# 이게 새로 짠 iTransformer encoder
class iTransformerEncoder(nn.Module):
    def __init__(self, obs_len, embedding_dim=128, h_dim=64, num_layers=1, dropout=0.1):
        super(iTransformerEncoder, self).__init__()

        self.obs_len = obs_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        # ✅ iTransformer의 DataEmbedding 사용
        self.enc_embedding = DataEmbedding_inverted(
            input_dim=2,
            expand_dim=64,  # (x, y) 좌표 입력
            d_model=self.embedding_dim,  # 128차원으로 변환
            # embed_type="timeF",
            # freq="s",
            dropout=dropout
        )

        # ✅ iTransformer의 Encoder 사용
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                        self.embedding_dim,  # 기존 d_model
                        n_heads=8
                    ),
                    self.embedding_dim,  # d_model
                    self.h_dim * 4,  # Feedforward 차원
                    dropout=dropout,
                    activation="gelu"
                ) for _ in range(self.num_layers)
            ],
            norm_layer=nn.LayerNorm(self.embedding_dim)  # iTransformer에서는 마지막에 LayerNorm 적용
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (seq_len, batch, 2)  # (x, y)

        Output:
        - final_h: Tensor of shape (batch, seq_len, embedding_dim)
        """
        batch_size = obs_traj.shape[1]
        seq_len = obs_traj.shape[0]

        # ✅ iTransformer는 변수 중심이므로 좌표(x, y)를 개별 변수로 인식
        obs_traj_T = obs_traj.permute(1, 2, 0)  # (batch, 2, seq_len)

        # ✅ iTransformer Embedding 적용
        obs_traj_embedded = self.enc_embedding(obs_traj_T, None)  # (batch, 2, embedding_dim)

        # ✅ iTransformer Encoder 실행
        enc_out, _ = self.encoder(obs_traj_embedded, attn_mask=None)  # (batch, 2, embedding_dim)

        return enc_out

class iTransformerTrafficEncoder(nn.Module):
    def __init__(self, traffic_state_dim=5, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(iTransformerTrafficEncoder, self).__init__()
        
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # ✅ 트래픽 상태를 위한 Embedding Layer (각 트래픽 상태를 embedding_dim으로 변환)
        self.traffic_embedding = nn.Embedding(traffic_state_dim, embedding_dim)

        # ✅ iTransformer의 Embedding 적용 (변수 중심 Attention)
        self.enc_embedding = DataEmbedding_inverted(
            traffic_state_dim, self.embedding_dim, dropout=dropout
        )

        # ✅ iTransformer의 Encoder 사용 (FullAttention 기반)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                        self.embedding_dim,  # 기존 d_model
                        n_heads=8
                    ),
                    self.embedding_dim,  # d_model
                    self.h_dim * 4,  # Feedforward 차원
                    dropout=dropout,
                    activation="gelu"
                ) for _ in range(self.num_layers)
            ],
            norm_layer=nn.LayerNorm(self.embedding_dim)  # iTransformer에서는 마지막에 LayerNorm 적용
        )

        # ✅ Projection Layer 추가 (Decoder 없이 바로 예측 가능하도록 설정)
        self.projection_layer = nn.Linear(self.embedding_dim, traffic_state_dim)

    def forward(self, traffic_state):
        """
        Inputs:
        - traffic_state: Tensor of shape (batch, traffic_state_dim)
        Output:
        - final_h: Tensor of shape (batch, traffic_state_dim, feature_dim)
        """
        batch_size = traffic_state.size(0)

        # ✅ 트래픽 상태를 Embedding 벡터로 변환
        traffic_state_embedding = self.traffic_embedding(traffic_state)  # (batch, traffic_state_dim, embedding_dim)

        # ✅ iTransformer의 Embedding 적용 (변수 기준 Attention)
        enc_out = self.enc_embedding(traffic_state_embedding, None)  # (batch, traffic_state_dim, embedding_dim)

        # ✅ iTransformer Encoder 실행
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (batch, traffic_state_dim, embedding_dim)

        # ✅ Projection Layer를 통해 최종 출력
        final_h = self.projection_layer(enc_out)  # (batch, traffic_state_dim, feature_dim)

        return final_h

class iTransformerVehicleEncoder(nn.Module):
    def __init__(self, agent_type_dim=6, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(iTransformerVehicleEncoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # ✅ 차량 유형을 위한 Embedding Layer
        self.agent_type_embedding = nn.Embedding(agent_type_dim, embedding_dim)

        # ✅ 차량 크기 정보를 임베딩 차원으로 변환하는 Linear Layer
        self.size_layer = nn.Linear(2, embedding_dim)

        # ✅ iTransformer의 Embedding 적용 (변수 중심 Attention)
        self.enc_embedding = DataEmbedding_inverted(
            embedding_dim * 2, self.embedding_dim, dropout=dropout
        )

        # ✅ iTransformer의 Encoder 사용 (FullAttention 기반)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                        self.embedding_dim,  # 기존 d_model
                        n_heads=8
                    ),
                    self.embedding_dim,  # d_model
                    self.h_dim * 4,  # Feedforward 차원
                    dropout=dropout,
                    activation="gelu"
                ) for _ in range(self.num_layers)
            ],
            norm_layer=nn.LayerNorm(self.embedding_dim)  # iTransformer에서는 마지막에 LayerNorm 적용
        )

        # ✅ Projection Layer 추가 (Decoder 없이 바로 예측 가능하도록 설정)
        self.projection_layer = nn.Linear(self.embedding_dim, embedding_dim * 2)

    def forward(self, agent_type, size):
        """
        Inputs:
        - agent_type: Tensor of shape (seq_len, batch) → 차량 유형 인덱스
        - size: Tensor of shape (seq_len, batch, 2) → 차량 크기 정보
        Output:
        - final_h: Tensor of shape (batch, 변수 개수, seq_len)
        """
        batch_size = agent_type.size(1)
        seq_len = agent_type.size(0)
    
        # ✅ 차량 타입을 Embedding 벡터로 변환
        agent_type = agent_type.to(dtype=torch.long, device=agent_type.device)
        agent_type_embedding = self.agent_type_embedding(agent_type)  # (seq_len, batch, embedding_dim)
    
        # ✅ 차량 크기 정보를 Embedding 차원으로 변환
        size_embedding = self.size_layer(size)  # (seq_len, batch, embedding_dim)
        
        # ✅ 크기 변환 확인
        print(f"size.shape: {size.shape}")
        print(f"size_embedding.shape before enc_embedding: {size_embedding.shape}")
    
        # 🚨 size_embedding의 크기 맞추기
        size_embedding = size_embedding.view(seq_len, batch_size, self.embedding_dim)  # (12, 769, 64)로 유지
    
        # ✅ enc_embedding 적용
        size_embedding = self.enc_embedding(size_embedding)  # (seq_len, batch, embedding_dim)
        
        print(f"size_embedding.shape after enc_embedding: {size_embedding.shape}")
    
        # 🚨 강제 차원 맞추기 (디버깅용)
        if size_embedding.shape[-1] != 128:
            print(f"⚠️ Warning: size_embedding.shape[-1] is {size_embedding.shape[-1]}, expected 128")
            size_embedding = size_embedding[:, :, :128]  # 강제 차원 맞추기
    
        # ✅ 두 정보를 결합하여 iTransformer 입력 형식으로 변환
        combined_embedding = torch.cat([agent_type_embedding, size_embedding], dim=-1)  # (seq_len, batch, 2 * embedding_dim)
    
        # ✅ iTransformer는 변수 중심이므로 차원 변환 (batch, 변수 개수, seq_len)
        combined_embedding_T = combined_embedding.permute(1, 2, 0)  # (batch, 2 * embedding_dim, seq_len)
    
        # ✅ iTransformer의 Embedding 적용
        enc_out = self.enc_embedding(combined_embedding_T, None)  # (batch, 2 * embedding_dim, seq_len)
    
        # ✅ iTransformer Encoder 실행
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (batch, 2 * embedding_dim, seq_len)
    
        # ✅ Projection Layer를 통해 최종 출력
        final_h = self.projection_layer(enc_out)  # (batch, 2 * embedding_dim, seq_len)
    
        return final_h

class iTransformerStateEncoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(iTransformerStateEncoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # ✅ 속도(vx, vy) 및 가속도(ax, ay)를 Embedding 차원으로 변환하는 Linear Layer
        self.state_layer = nn.Linear(4, embedding_dim)  # (vx, vy, ax, ay → embedding_dim)

        # ✅ iTransformer의 Embedding 적용 (변수 중심 Attention)
        self.enc_embedding = DataEmbedding_inverted(
            embedding_dim, self.embedding_dim, dropout=dropout
        )

        # ✅ iTransformer의 Encoder 사용 (FullAttention 기반)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                        self.embedding_dim,  # 기존 d_model
                        n_heads=8
                    ),
                    self.embedding_dim,  # d_model
                    self.h_dim * 4,  # Feedforward 차원
                    dropout=dropout,
                    activation="gelu"
                ) for _ in range(self.num_layers)
            ],
            norm_layer=nn.LayerNorm(self.embedding_dim)  # iTransformer에서는 마지막에 LayerNorm 적용
        )

        # ✅ Projection Layer 추가 (Decoder 없이 바로 예측 가능하도록 설정)
        self.projection_layer = nn.Linear(self.embedding_dim, embedding_dim)

    def forward(self, state):
        """
        Inputs:
        - state: Tensor of shape (seq_len, batch, 4)  # (vx, vy, ax, ay)

        Output:
        - final_h: Tensor of shape (batch, 변수 개수=4, seq_len)
        """
        batch_size = state.size(1)
        seq_len = state.size(0)

        # ✅ 속도 및 가속도를 Embedding 차원으로 변환
        state_embedding = self.state_layer(state)  # (seq_len, batch, embedding_dim)

        # ✅ iTransformer는 변수 중심이므로 차원 변환 (batch, 변수 개수, seq_len)
        state_embedding_T = state_embedding.permute(1, 2, 0)  # (batch, embedding_dim, seq_len)

        # ✅ iTransformer의 Embedding 적용
        enc_out = self.enc_embedding(state_embedding_T, None)  # (batch, embedding_dim, seq_len)

        # ✅ iTransformer Encoder 실행
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (batch, embedding_dim, seq_len)

        # ✅ Projection Layer를 통해 최종 출력
        final_h = self.projection_layer(enc_out)  # (batch, embedding_dim, seq_len)

        return final_h

#새로 추가한 iTransformer를 사용하는 Decoder
class iTransformerDecoder(nn.Module):
    def __init__(self, pred_len, embedding_dim=64, h_dim=128, mlp_dim=1024, dropout=0.0):
        super(iTransformerDecoder, self).__init__()

        self.pred_len = pred_len  # 예측할 길이 (ex. 12)
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim

        # ✅ iTransformer는 Decoder가 없으므로 Projection Layer 활용
        self.projection_layer = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, pred_len * 2)  # 최종 예측 좌표 (x, y) 변환
        )

    def forward(self, encoded_features):
        """
        Inputs:
        - encoded_features: (batch, feature_dim)  
          iTransformer Encoder에서 추출된 Feature들 (Trajectory, Traffic, Vehicle, State Encoding 포함)

        Output:
        - pred_traj_fake_rel: (pred_len, batch, 2) → 예측된 상대 경로
        """
        batch = encoded_features.size(0)

        # ✅ Projection Layer를 통해 최종 예측 (Encoder의 Output을 직접 변환)
        pred_traj_fake_rel = self.projection_layer(encoded_features).view(self.pred_len, batch, 2)

        return pred_traj_fake_rel

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
class iTransformerTrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, dropout=0.0,
        traffic_h_dim=64
    ):
        super(iTransformerTrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.traffic_h_dim = traffic_h_dim

        # ✅ 기존 TransformerEncoder 대신 iTransformerEncoder 사용
        self.encoder = iTransformerEncoder(
            obs_len=obs_len,
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # ✅ 기존 Traffic, Vehicle, State Encoder를 iTransformer 기반으로 변경
        self.traffic_encoder = iTransformerTrafficEncoder(traffic_state_dim=5, embedding_dim=64, h_dim=traffic_h_dim)
        self.vehicle_encoder = iTransformerVehicleEncoder(agent_type_dim=6, embedding_dim=64, h_dim=64)
        self.state_encoder = iTransformerStateEncoder(embedding_dim=embedding_dim, h_dim=64)

        # ✅ 기존 TransformerDecoder 대신 iTransformerDecoder 사용
        self.decoder = iTransformerDecoder(
            pred_len=pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state, user_noise=None):
        batch = obs_traj_rel.size(1)
    
        # ✅ 기존 Transformer: (seq_len, batch, 변수) → iTransformer: (batch, 변수, seq_len) 변환
        obs_traj_rel_T = obs_traj_rel.permute(1, 2, 0)  # (batch, 2, seq_len)
    
        # ✅ iTransformer 기반 Encoder → Feature 추출
        final_encoder_h = self.encoder(obs_traj_rel_T)  # (batch, 변수, obs_len)
    
        # ✅ 추가적인 Feature Encoding (Traffic, Vehicle, State)
        vehicle_encoding = self.vehicle_encoder(agent_type, size).permute(1, 2, 0)  # (batch, 변수, obs_len)
        state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2)).permute(1, 2, 0)  # (batch, 변수, obs_len)
        traffic_encoding = self.traffic_encoder(traffic_state).permute(1, 2, 0)  # (batch, 변수, obs_len)
    
        # ✅ iTransformer에서는 Pooling 없이 Attention으로 모든 정보를 직접 학습
        combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding, traffic_encoding], dim=1)  # (batch, 모든 변수, obs_len)
    
        # ✅ Noise 추가 (필요한 경우)
        combined_encoding = self.add_noise(combined_encoding, seq_start_end, user_noise)
    
        # ✅ Projection Layer를 사용하여 Decoder 역할 수행
        pred_traj_fake_rel = self.decoder(combined_encoding)
    
        return pred_traj_fake_rel

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (batch, feature_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
          relation between different types of noise and outputs.
        Outputs:
        - output: Tensor of shape (batch, feature_dim) with added noise
        """
        if self.noise_dim[0] == 0:
            return _input

        noise_shape = (_input.size(0),) + self.noise_dim
        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        return torch.cat([_input, z_decoder], dim=1)

    def mlp_decoder_needed(self):
        """
        기존 Transformer 기반에서는 MLP Decoder가 필요했지만,  
        iTransformer 구조에서는 Projection Layer를 사용하므로 불필요.
        """
        return False  # iTransformer 기반에서는 필요 없음

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state, user_noise=None):
        batch = obs_traj_rel.size(1)

        # ✅ iTransformer 기반 Encoder → Feature 추출 (변수 중심 Attention)
        final_encoder_h = self.encoder(obs_traj_rel)  # (batch, 변수, obs_len)

        # ✅ 추가적인 Feature Encoding (Traffic, Vehicle, State)
        vehicle_encoding = self.vehicle_encoder(agent_type, size)  # (batch, 변수, obs_len)
        state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))  # (batch, 변수, obs_len)
        traffic_encoding = self.traffic_encoder(traffic_state)  # (batch, 변수, obs_len)

        # ✅ Feature 차원을 obs_len(12)과 맞추기
        vehicle_encoding = vehicle_encoding.expand(batch, -1, self.obs_len)  # (batch, 변수, obs_len)
        state_encoding = state_encoding.expand(batch, -1, self.obs_len)  # (batch, 변수, obs_len)
        traffic_encoding = traffic_encoding.expand(batch, -1, self.obs_len)  # (batch, 변수, obs_len)

        # ✅ iTransformer에서는 Pooling 없이 Attention으로 모든 정보를 직접 학습
        combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding, traffic_encoding], dim=1)  # (batch, 모든 변수, obs_len)

        # ✅ Noise 추가 (필요한 경우)
        combined_encoding = self.add_noise(combined_encoding, seq_start_end, user_noise)

        # ✅ Projection Layer를 사용하여 Decoder 역할 수행
        pred_traj_fake_rel = self.decoder(combined_encoding)

        return pred_traj_fake_rel

# 이거는 새로 짠 discriminator(transformer 사용한 것)
class iTransformerTrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(iTransformerTrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.embedding_dim = embedding_dim

        self.spatial_embedding = nn.Linear(2, embedding_dim)

        # ✅ 기존 Transformer Encoder 대신 iTransformerEncoder 사용
        self.encoder = iTransformerEncoder(
            obs_len=self.seq_len,
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # ✅ iTransformer 기반 분류기 정의
        real_classifier_dims = [h_dim, mlp_dim, 1]
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

        # ✅ 2D 위치 데이터를 iTransformer 입력 차원으로 변환
        traj_rel_embedded = self.spatial_embedding(traj_rel)  # (seq_len, batch, embedding_dim)

        # ✅ iTransformer Encoder 적용
        final_h = self.encoder(traj_rel_embedded)  # (batch, feature_dim, seq_len)

        # ✅ iTransformer는 변수 중심이므로 최종 Feature를 평균 Pooling 적용
        classifier_input = final_h.mean(dim=2)  # (batch, feature_dim)

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
