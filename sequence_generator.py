import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class SequenceGenerator:
    def __init__(self, model_name="microsoft/phi-1_5"):
        # Initialize device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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

    def generate_next_token_probs(self, sequences):
        """
        Takes in a batch of sequences and computes the probabilities of the next token for each sequence.
        """
        # Ensure input is on the correct device
        sequences = sequences.to(self.device)
        
        # Calculate logits for the next token
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(input_ids=sequences)
            next_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
        
        return probs
    
    