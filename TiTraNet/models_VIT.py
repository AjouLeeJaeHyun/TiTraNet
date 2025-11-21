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
    
class ViTMapEncoder(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=3, embed_dim=64, num_layers=1, num_heads=4, dropout=0.1):
        super(ViTMapEncoder, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Flatten patch and project to embedding dim
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

        # Positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling output (mean over tokens)
        self.pool = nn.AdaptiveAvgPool1d(1)

    # 기존 forward(VIT)
    # def forward(self, map_image):
    #     B, C, H, W = map_image.shape
    #     patch_H, patch_W = self.patch_size, self.patch_size

    #     # Step 1: Divide image into patches
    #     patches = map_image.unfold(2, patch_H, patch_H).unfold(3, patch_W, patch_W)  # (B, C, n_patches_H, n_patches_W, patch_H, patch_W)
    #     patches = patches.contiguous().view(B, C, -1, patch_H, patch_W)  # (B, C, N_patches, pH, pW)
    #     patches = patches.permute(0, 2, 1, 3, 4)  # (B, N_patches, C, pH, pW)
    #     patches = patches.flatten(2)  # (B, N_patches, C*pH*pW)

    #     # Step 2: Linear projection + position embedding
    #     tokens = self.patch_embed(patches) + self.pos_embed  # (B, N_patches, D)

    #     # Step 3: Transformer encoder
    #     tokens = tokens.permute(1, 0, 2)  # (N_patches, B, D)
    #     encoded = self.transformer(tokens)  # (N_patches, B, D)
    #     encoded = encoded.permute(1, 2, 0)  # (B, D, N_patches)

    #     # Step 4: Global pooling over patches
    #     pooled = self.pool(encoded).squeeze(-1)  # (B, D)
    #     return pooled

    def forward(self, map_image):
        B, C, H, W = map_image.shape
        patch_H, patch_W = self.patch_size, self.patch_size

        # Step 1: Divide image into patches
        patches = map_image.unfold(2, patch_H, patch_H).unfold(3, patch_W, patch_W)
        patches = patches.contiguous().view(B, C, -1, patch_H, patch_W)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.flatten(2)  # (B, N_patches, C*pH*pW)

        # Step 2: Linear projection + position embedding
        tokens = self.patch_embed(patches) + self.pos_embed  # (B, N_patches, D)

        # Step 3: Transformer encoder
        tokens = tokens.permute(1, 0, 2)  # (N_patches, B, D)
        encoded = self.transformer(tokens)  # (N_patches, B, D)
        encoded = encoded.permute(1, 0, 2)  # ✅ (B, N_patches, D)

        return encoded  # ✅ 전체 patch token을 반환
    
# 기존 VIT에서 trajectory2map cross attention layer 추가한 것
class TrajMapCrossAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, traj_feat, map_feat):
        """
        traj_feat: [B, T, D]
        map_feat: [B, N_map, D]
        """
        # Q: trajectory encoder output (T), K/V: map patches (N)
        attn_output, _ = self.attn(traj_feat, map_feat, map_feat)  # (B, T, D)

        # Residual + FFN
        x = self.norm(traj_feat + attn_output)
        return self.ffn(x)

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
        traffic_h_dim=64, map_feature_dim=64
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.traffic_h_dim = traffic_h_dim
        self.traj_map_attn = TrajMapCrossAttention(
            d_model=embedding_dim,  # ViT의 embed_dim과 동일해야 함
            nhead=4,
            dropout=0.1
        )

        self.encoder = TransformerEncoder(
            obs_len=obs_len,
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.traffic_encoder = TrafficEncoder(traffic_state_dim=5, embedding_dim=64, h_dim=traffic_h_dim)
        self.vehicle_encoder = VehicleEncoder(agent_type_dim=6, embedding_dim=64, h_dim=64)
        self.state_encoder = StateEncoder(embedding_dim=embedding_dim, h_dim=64)
        self.map_encoder = ViTMapEncoder(image_size=128, patch_size=16, in_channels=3, embed_dim=128, num_layers=2, num_heads=4)

        self.decoder = TransformerDecoder(
            seq_len=pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def add_noise(self, _input, seq_start_end, user_noise=None):
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
    def forward(self, obs_traj, obs_traj_rel, seq_start_end,
                vx, vy, ax, ay, agent_type, size, traffic_state,
                map_image, user_noise=None):

        batch = obs_traj_rel.size(1)

        final_encoder_h = self.encoder(obs_traj_rel)
        vehicle_encoding = self.vehicle_encoder(agent_type, size)
        state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))
        traffic_encoding = self.traffic_encoder(traffic_state)

        map_encoding = self.map_encoder(map_image)
        # print("map_encoding.shape = ", map_encoding.shape)
        map_encoding = map_encoding.to(obs_traj_rel.device)

        traj_feat = final_encoder_h.permute(1, 0, 2)  # (B, T, D)
        map_feat = map_encoding  # (B, N_map, D)

        # Cross-attn
        map_context = self.traj_map_attn(traj_feat, map_feat)  # (B, T, D)

        # 다시 (T, B, D)로 변환
        map_context = map_context.permute(1, 0, 2)

        if len(vehicle_encoding.shape) == 2:
            vehicle_encoding = vehicle_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)
        if len(state_encoding.shape) == 2:
            state_encoding = state_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)
        if len(traffic_encoding.shape) == 2:
            traffic_encoding = traffic_encoding.unsqueeze(0).expand(self.obs_len, -1, -1)

        combined_encoding = torch.cat([
            final_encoder_h,
            vehicle_encoding,
            state_encoding,
            traffic_encoding,
            map_context,
        ], dim=2)

        # dynamic feature dim 계산
        total_feature_dim = (
            final_encoder_h.size(2) +
            vehicle_encoding.size(2) +
            state_encoding.size(2) +
            traffic_encoding.size(2) +
            map_context.size(2)
        )

        # print("total_feature_dim =", total_feature_dim)

        # projection layer가 없을 때만 초기화 (보통 __init__이 아니라 forward에서 동적으로 처리할 때 사용)
        if not hasattr(self, 'memory_projection'):
            self.memory_projection = nn.Linear(total_feature_dim, self.embedding_dim).to(final_encoder_h.device)

        memory = self.memory_projection(combined_encoding)

        pred_traj_fake_rel = self.decoder(
            last_pos=obs_traj[-1],
            last_pos_rel=obs_traj_rel[-1],
            memory=memory,
            seq_start_end=seq_start_end,
            vx=vx, vy=vy
        )

        return pred_traj_fake_rel

# 이거는 새로 짠 discriminator(transformer 사용한 것)
class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local', map_feature_dim=64
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.embedding_dim = embedding_dim
        self.map_feature_dim = map_feature_dim

        self.spatial_embedding = nn.Linear(2, embedding_dim)

        # ✅ ViT-based Map Encoder
        self.map_encoder = ViTMapEncoder(
            image_size=128,
            patch_size=16,
            in_channels=3,
            embed_dim=map_feature_dim,
            num_layers=4,
            num_heads=4
        )

        # Positional Encoding for trajectory
        self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, 1, embedding_dim))

        # Transformer Encoder over combined trajectory+map
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim + map_feature_dim,
            nhead=8,
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Classifier
        real_classifier_dims = [embedding_dim + map_feature_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(self, traj, traj_rel, map_data, seq_start_end=None):
        """
        Inputs:
        - traj_rel: (seq_len, batch, 2)
        - map_data: (batch, 3, H, W)
        """
        batch = traj_rel.shape[1]

        # 🧍 Trajectory embedding
        pos_enc = self.positional_encoding[:traj_rel.size(0), :, :] # safe slicing
        traj_rel_embedded = self.spatial_embedding(traj_rel) + pos_enc  # (seq_len, B, embed)

        # 🗺️ Map encoding with ViT
        map_feature = self.map_encoder(map_data)  # (B, map_feature_dim)
        map_feature = map_feature.mean(dim=1)
        map_feature = map_feature.unsqueeze(0).expand(self.seq_len, -1, -1)

        # 🧩 Combine
        combined = torch.cat([traj_rel_embedded, map_feature], dim=2)  # (seq_len, B, embed + map_feat)

        # 🔀 Transformer encoder
        encoded = self.encoder(combined)  # (seq_len, B, d_model)

        # 🧠 Classify using last time step
        last_h = encoded[-1]  # (B, d_model)
        score = self.real_classifier(last_h)  # (B, 1)

        return score