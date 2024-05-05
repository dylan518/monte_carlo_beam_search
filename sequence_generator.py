import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class SequenceGenerator:
    def __init__(self, model_name="microsoft/phi-1_5"):
        # Initialize device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
                # Add padding token to the tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Utilize multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
    
    def tokenize_string(self, string):
        """
        Tokenizes a string using the model's tokenizer.
        """
        return self.tokenizer(string, return_tensors="pt", padding=True, truncation=True)["input_ids"]

    def generate_next_token_probs(self, sequences, top_n=5):
        """
        Takes in a batch of sequences and computes the probabilities of the next token for each sequence in the batch.
        Returns a list of dictionaries for each sequence, where each dictionary has tokens as keys and probabilities as values.
        """
        # Ensure input is on the correct device
        sequences = sequences.to(self.device)

        # Calculate logits for the next token
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(input_ids=sequences)
            next_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

        # Find the top_n probabilities and their indices
        top_probs, top_indices = torch.topk(probs, top_n, dim=1)

        # Decode each index to its corresponding token and create the dictionary
        results = []
        for i in range(top_indices.shape[0]):
            token_probs_dict = {}
            for j in range(top_indices.shape[1]):
                token = self.tokenizer.decode([top_indices[i, j].item()])
                probability = top_probs[i, j].item()
                token_probs_dict[token] = probability
            results.append(token_probs_dict)

        return results
    
    