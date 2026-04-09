import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# ----------------------------
# 1. Small dataset
# ----------------------------
sentences = [
    "i love deep learning",
    "transformers are powerful",
    "attention is all you need",
    "i enjoy coding",
    "deep learning is fun",
    "nlp is interesting",
    "pytorch makes it easy",
    "i love transformers",
    "models learn patterns",
    "data drives models"
]

# ----------------------------
# 2. Tokenization
# ----------------------------
tokens = [s.split() for s in sentences]
vocab = sorted(set(word for sent in tokens for word in sent))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

max_len = max(len(s) for s in tokens)

def encode(sent):
    ids = [word2idx[w] for w in sent]
    ids += [0]*(max_len - len(ids))  # padding
    return ids

input_ids = torch.tensor([encode(s) for s in tokens])

print("Input tokens:\n", input_ids)

# ----------------------------
# 3. Positional Encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/d_model)))
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ----------------------------
# 4. Mini Transformer Encoder
# ----------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, heads=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)

        attn_output, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + attn_output)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x, attn_weights

# ----------------------------
# Run model
# ----------------------------
model = MiniTransformer(len(vocab), d_model=64, heads=2)

output, attn_weights = model(input_ids)

# ----------------------------
# 5. Outputs
# ----------------------------
print("\nFinal contextual embeddings (shape):", output.shape)

# Show embeddings for first sentence
print("\nEmbeddings for first sentence:\n", output[0])

# ----------------------------
# Attention Heatmap (first sentence)
# ----------------------------
import matplotlib.pyplot as plt

attn = attn_weights[0].detach()

plt.imshow(attn, cmap='viridis')
plt.colorbar()

words = tokens[0] + ["<pad>"]*(max_len - len(tokens[0]))
plt.xticks(range(max_len), words, rotation=45)
plt.yticks(range(max_len), words)

plt.title("Attention Heatmap (Sentence 1)")
plt.show()
