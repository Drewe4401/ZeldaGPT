import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 32
eval_interval = 500
eval_iters = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(300)

with open('zelda_text_dump.txt', 'r', encoding='utf-8') as f:
  text = f.read()
  
chars_in_text = sorted(list(set(text)))
vocab_size = len(chars_in_text)
stoi = { ch:i for i,ch in enumerate(chars_in_text) }
itos = { i:ch for i,ch in enumerate(chars_in_text) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(.9*len(data))
training_data = data[:n] # 90% of of text will be training data
validation_data = data[n:] # 10% of text will be validation data

def get_batch(split):
  data = training_data if split == 'train' else validation_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) # random block generation in a 4x8
  x = torch.stack([data[i:i+block_size] for i in ix]) # current integer
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # next integer
  x, y = x.to(device), y.to(device)
  return x, y


# Use the torch.no_grad() decorator to indicate that the following function
# should be executed without calculating gradients, as it is used for evaluation
@torch.no_grad()
def estimate_loss():
    # Initialize an empty dictionary to store the output losses for different data splits
    out = {}
    
    # Set the model to evaluation mode to disable dropout and other training-specific layers
    model.eval()
    
    # Loop over the data splits ('train' and 'val') to estimate the loss for each split
    for split in ['train', 'val']:
        # Initialize a tensor of zeros to store the individual losses for each evaluation iteration
        losses = torch.zeros(eval_iters)
        
        # Loop for the specified number of evaluation iterations
        for k in range(eval_iters):
            # Get a batch of data (input and target) for the current split using the get_batch function
            X, Y = get_batch(split)
            
            # Perform the forward pass on the input data (X) and target data (Y)
            # and get the output logits and the loss
            logits, loss = model(X, Y)
            
            # Store the current iteration's loss in the losses tensor
            losses[k] = loss.item()
        
        # Calculate the mean loss for the current split and store it in the output dictionary
        out[split] = losses.mean()
    
    # Set the model back to training mode to enable dropout and other training-specific layers
    model.train()
    
    # Return the output dictionary containing the mean losses for each data split
    return out


# Define the BigramLanguageModel class, which inherits from the PyTorch nn.Module class
class BigramLanguageModel(nn.Module):
    # Initialize the BigramLanguageModel class
    def __init__(self, vocab_size):
        # Call the parent class constructor
        super(BigramLanguageModel, self).__init__()
        # Create an embedding layer with vocab_size input and output dimensions
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # Define the forward pass for the model
    def forward(self, idx, targets=None):
        # Calculate the logits by passing the input idx through the embedding layer
        logits = self.token_embedding_table(idx)  # (B, T, C) (batch_size, block_size, vocab_size)

        # If there are no targets, set the loss to None
        if targets is None:
            loss = None
        else:
            # If there are targets, reshape the logits and targets for the loss calculation
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Calculate the cross-entropy loss between logits and targets
            loss = F.cross_entropy(logits, targets)

        # Return both logits and loss
        return logits, loss

    # Define the generate function for generating text
    def generate(self, idx, max_new_tokens):
        # Loop for the specified number of tokens to generate
        for _ in range(max_new_tokens):
            # Calculate the logits and loss
            logits, loss = self(idx)
            # Take the last token logits and calculate the softmax probabilities
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # Sample the next token index based on the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            # Concatenate the sampled token index with the existing indices
            idx = torch.cat((idx, idx_next), dim=1)

        # Return the generated token indices
        return idx
    
# Instantiate the BigramLanguageModel with vocab_size
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create an optimizer for the model 'm'
# Use the AdamW optimization algorithm with a learning rate of 1e-2 (0.01)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(10000):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# Generate a sequence of tokens using the trained model 'm'
# Pass an initial input tensor of zeros with a shape of (1, 1) and dtype 'long'
# Set the number of tokens to generate to 500
generated_sequence = m.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)

# Convert the generated token tensor to a list of token IDs
generated_token_ids = generated_sequence[0].tolist()

# Decode the list of token IDs into human-readable text using the 'decode' function
decoded_text = decode(generated_token_ids)

# Print the decoded text
print(decoded_text)