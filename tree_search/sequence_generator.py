# sequence_generator.py

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class SequenceGenerator:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct", ddp_model=False, rank=0):
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model '{model_name}' on device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_string(self, string):
        tokenized_output = self.tokenizer(
            string,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=False
        )["input_ids"].to(self.device)
        return tokenized_output

    def token_to_string(self, token_indices):
        if isinstance(token_indices, torch.Tensor):
            token_indices = token_indices.squeeze()
            token_indices_list = token_indices.tolist()
        else:
            token_indices_list = token_indices

        decoded_string = self.tokenizer.decode(token_indices_list, skip_special_tokens=True)
        return decoded_string

    def generate_next_token_probs(self, sequences, top_n=5, temperature=1.0):
        sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=sequences)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, top_n, dim=1)

        return top_probs, top_indices


