import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sequence_generator import SequenceGenerator


class TestSequenceGenerator(unittest.TestCase):
    def setUp(self):
        self.model_name = "microsoft/Phi-1.5"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_sequence_generator(self):
        if self.device.type == "cuda":
            print("Running tests on GPU.")
            sequence_generator = SequenceGenerator(self.model_name)

            # Test tokenization
            input_string = "Hello, how are you?"
            tokenized_string = sequence_generator.tokenize_string(input_string)
            self.assertIsInstance(tokenized_string, torch.Tensor)
            self.assertEqual(tokenized_string.shape[0], 1)  # Batch size should be 1
            self.assertGreater(tokenized_string.shape[1], 0)  # Length should be greater than 0

            # Test next token probabilities
            sequences = tokenized_string
            probs = sequence_generator.generate_next_token_probs(sequences)
            self.assertIsInstance(probs, torch.Tensor)
            self.assertEqual(probs.shape[0], 1)  # Batch size should be 1
            self.assertGreater(probs.shape[1], 0)  # Vocabulary size should be greater than 0
            self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.tensor([1.0])))  # Probabilities should sum to 1

            # Generate some sample output
            print("Sample output:")
            output_tokens = sequences[0].tolist()
            for _ in range(10):  # Generate 10 more tokens
                next_token = torch.multinomial(probs[0], num_samples=1)
                output_tokens.append(next_token.item())
                probs = sequence_generator.generate_next_token_probs(torch.tensor([output_tokens]))
            
            output_string = sequence_generator.tokenizer.decode(output_tokens)
            print(output_string)
        else:
            print("No GPU found. Skipping tests.")
            self.skipTest("No GPU found.")


if __name__ == '__main__':
    unittest.main()