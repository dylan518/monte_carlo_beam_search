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
        return self.tokenizer(string, return_tensors="pt", padding=True, truncation=True, return_attention_mask=False)["input_ids"]
    
    def decode_token_tensor(self,token_indices_tensor):
      """
      Decodes a tensor of token indices into their corresponding token strings. This function
      is designed to handle tensors that might be on the GPU.

      Args:
          token_indices_tensor (torch.Tensor): Tensor of token indices, possibly on GPU.
          tokenizer: Tokenizer object with a convert_ids_to_tokens method.

      Returns:
          list of str: List of decoded tokens.
      """
      # Move the tensor to the CPU if it's on the GPU
      if token_indices_tensor.is_cuda:
          token_indices_tensor = token_indices_tensor.cpu()
      
      # Convert the tensor to a list of integers
      token_indices_list = token_indices_tensor.tolist()

      # Decode each token index to a token string using the tokenizer
      decoded_tokens = [self.tokenizer.convert_ids_to_tokens(index) for index in token_indices_list]

      return decoded_tokens
    
    def token_to_string(self, token_indices_list):
        decoded_string=[self.tokenizer.convert_ids_to_tokens(index) for index in token_indices_list]
        return self.reconstruct_sentence(decoded_string)
    
    def generate_next_token_probs(self, sequences, top_n=5):
        """
        Takes in a batch of sequences and computes the probabilities of the next token for each sequence in the batch.
        Returns the top_n probabilities and their indices for each sequence.
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

        return top_probs, top_indices
    
    def reconstruct_sentence(tokens):
      """
      Reconstructs a sentence from tokens encoded with special characters like 'Ġ'.
      
      Args:
          tokens (list of str): The list of tokens to be reconstructed into a sentence.

      Returns:
          str: The reconstructed sentence.
      """
      sentence = ""
      for token in tokens:
          if token.startswith('Ġ'):
              if sentence:  # Add space before if it's not the first token
                  sentence += ' '
              sentence += token[1:]  # Add the token without 'Ġ'
          else:
              sentence += token  # Add other tokens directly
      return sentence
    
    

