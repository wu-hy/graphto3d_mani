import torch
import torch.nn.functional as F

def get_next_token_probabilities(model, tokenizer, text):
    # 1. Encode input text
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # 2. Forward pass through the model (no gradient needed)
    with torch.no_grad():
        outputs = model(input_ids)
        # outputs.logits has shape [batch_size, sequence_length, vocab_size]
    
    # 3. Take the logits from the last token in the sequence
    next_token_logits = outputs.logits[:, -1, :]
    
    # 4. Convert logits to probabilities
    next_token_probs = F.softmax(next_token_logits, dim=-1)
    
    # next_token_probs will be shape [batch_size, vocab_size]
    # For a single input, that's [1, vocab_size], so you can flatten:
    next_token_probs = next_token_probs[0]  # shape: [vocab_size]
    
    # 5. Get token IDs and their probabilities
    # Convert to list of (token_id, probability) and sort by probability descending
    token_probs = [
        (token_id, float(prob))
        for token_id, prob in enumerate(next_token_probs)
    ]
    token_probs.sort(key=lambda x: x[1], reverse=True)

    return token_probs

def get_next_token_rank(model, tokenizer, text, next_token):
    next_token_id = tokenizer.encode(next_token, return_tensors='pt')
    token_probs = get_next_token_probabilities(model, tokenizer, text)
    
    # Also samples the next token here for seed up, choosing the first prob here:
    generated_id = 0
    # Find the rank of the next token
    for rank, (token_id, prob) in enumerate(token_probs):
        if rank == 0:
            generated_id = token_id
        if token_id == next_token_id:
            return rank + 1, generated_id  # Rank is 1-based
    
    raise NameError()  # If the token is not found

