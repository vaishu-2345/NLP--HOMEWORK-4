import torch
import torch.nn.functional as F
import math

# ----------------------------
# Attention Function
# ----------------------------
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)

    # Compute scores
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Without scaling (for comparison)
    softmax_no_scale = F.softmax(scores, dim=-1)

    # With scaling
    scaled_scores = scores / math.sqrt(d_k)
    attention_weights = F.softmax(scaled_scores, dim=-1)

    # Output
    output = torch.matmul(attention_weights, V)

    return scores, softmax_no_scale, attention_weights, output

# ----------------------------
# Test with random inputs
# ----------------------------
torch.manual_seed(0)

batch_size = 1
seq_len = 4
d_k = 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

scores, softmax_no_scale, attn_weights, output = scaled_dot_product_attention(Q, K, V)

# ----------------------------
# Print Results
# ----------------------------
print("\nRaw Scores (QK^T):\n", scores)

print("\nSoftmax WITHOUT scaling:\n", softmax_no_scale)

print("\nAttention Weights (WITH scaling):\n", attn_weights)

print("\nOutput Vectors:\n", output)

# ----------------------------
# Stability Check
# ----------------------------
print("\n--- Stability Check ---")
print("Max score before scaling:", scores.max().item())
print("Max score after scaling:", (scores / math.sqrt(d_k)).max().item())
