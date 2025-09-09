import torch
from vector_quantize_pytorch import ResidualVQ
import torch.nn as nn
from torch.nn.functional import cosine_similarity

class ResidualVectorQuanize(nn.Module):
    def __init__(self, input_dim, dim, codebook_dim, n_q, bins): # in_dim - dimension of the codebook emb, bins - number of embeddings per CB
        super().__init__()
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.n_q = n_q
        self.bins = bins

        # Model
        self.rvq_quantizer = ResidualVQ(
                dim = input_dim,
                codebook_size = bins, # codebook size
                decay = 0.9, # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 1.,   # the weight on the commitment loss
                threshold_ema_dead_code = 2,
                use_cosine_sim = False,
                codebook_dim = dim,
                num_quantizers= self.n_q
            )
        
    def cosine_sim_loss(self, features, target_features):
        assert features.shape == target_features.shape, "Shapes are missmatched"
        cosine_sim = cosine_similarity(features, target_features)
        loss = - torch.log(
            torch.sigmoid(cosine_sim)
        ).mean()
        return loss
    
    def forward(self, features):
        quantized, indices, commit_loss = self.rvq_quantizer(features)

    def encode(self, features):
        # B, T, C
        quantized, indices, commit_loss = self.rvq_quantizer(features)
        return indices

    def decode(self, codes):
        quantized=self.rvq_quantizer.get_output_from_indices(codes)
        return quantized


