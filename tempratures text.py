Python 3.8.3 (tags/v3.8.3:6f8c832, May 13 2020, 22:20:19) [MSC v.1925 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import torch
import torch.nn as nn
import torch.optim as optim
import random

# ---- Toy dataset ----
text = "hello help hello hero hello world help me hello there " * 50

chars = sorted(list(set(text)))
vocab_size = len(chars)

char2idx = {ch:i for i,ch in enumerate(chars)}
idx2char = {i:ch for ch,i in char2idx.items()}

data = [char2idx[c] for c in text]

# ---- Hyperparameters ----
seq_len = 50
hidden_size = 128
epochs = 10

# ---- Model ----
class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.rnn = nn.LSTM(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

model = CharRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# ---- Training ----
def get_batch():
    i = random.randint(0, len(data) - seq_len - 1)
    x = data[i:i+seq_len]
    y = data[i+1:i+seq_len+1]
    return torch.tensor(x), torch.tensor(y)

print("Training started...\n")

for epoch in range(epochs):
    total_loss = 0
    h = None

    for _ in range(200):
        x, y = get_batch()
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        optimizer.zero_grad()
        out, h = model(x, h)
        h = (h[0].detach(), h[1].detach())

        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/200:.4f}")

# ---- Text Generation ----
def sample(start="h", length=200, temp=1.0):
    model.eval()
    result = [start]
    input = torch.tensor([[char2idx[start]]])
    h = None

    for _ in range(length):
        out, h = model(input, h)
        logits = out[0, -1] / temp
        probs = torch.softmax(logits, dim=0).detach().numpy()

        idx = random.choices(range(vocab_size), weights=probs)[0]
        result.append(idx2char[idx])
        input = torch.tensor([[idx]])

    return "".join(result)

# ---- Generate outputs for all temperatures ----
print("\n\n--- Generated Text Outputs ---\n")

temps = [0.7, 1.0, 1.2]

for t in temps:
    print(f"\nTemperature = {t}\n")
    print(sample(temp=t))
    print("\n" + "-"*60)
