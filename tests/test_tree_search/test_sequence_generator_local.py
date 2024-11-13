import unittest
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer
from sequence_generator import SequenceGenerator
import torch
class TestSequenceGenerator(unittest.TestCase):
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def setUp(self, mock_tokenizer, mock_model):
        self.model_name = "microsoft/phi-1_5"
        self.sequence_generator = SequenceGenerator(self.model_name)
        self.mock_tokenizer = mock_tokenizer
        self.mock_model = mock_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_init(self):
        self.assertEqual(self.sequence_generator.device.type, "cpu")  # Assuming running on CPU for testing
        self.mock_tokenizer.assert_called_once_with(self.model_name)
        self.mock_model.assert_called_once_with(self.model_name, torch_dtype=torch.float32)

    def test_tokenize_string(self):
        string = "Hello, world!"
        expected_output = torch.tensor([[1, 2, 3]])  # Mocked tokenized output
        self.mock_tokenizer.return_value.return_value = {"input_ids": expected_output}

        tokenized_string = self.sequence_generator.tokenize_string(string)

        self.mock_tokenizer.return_value.assert_called_once_with(string, return_tensors="pt", padding=True, truncation=True)
        self.assertTrue(torch.equal(tokenized_string, expected_output))

    def test_generate_next_token_probs(self):
        if self.device.type == "cuda":
            print("Running tests on GPU.")
            sequence_generator = SequenceGenerator(self.model_name)

            # Test next token probabilities
            sequences = torch.randint(0, 1000, (2, 5))  # Example sequences
            with unittest.mock.patch.object(F, 'softmax', wraps=F.softmax) as mock_softmax:
                probs = sequence_generator.generate_next_token_probs(sequences)
                self.assertIsInstance(probs, torch.Tensor)
                self.assertEqual(probs.shape[0], 2)  # Batch size should be 2
                self.assertGreater(probs.shape[1], 0)  # Vocabulary size should be greater than 0
                self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.tensor([1.0, 1.0])))  # Probabilities should sum to 1

                # Check if softmax was called with the correct arguments
                mock_softmax.assert_called_once()
                call_args = mock_softmax.call_args[0]
                self.assertTrue(torch.allclose(call_args[0], sequence_generator.model(input_ids=sequences).logits[:, -1, :]))
                self.assertEqual(call_args[1], -1)
        else:
            print("No GPU found. Skipping tests.")
            self.skipTest("No GPU found.")


if __name__ == '__main__':
    unittest.main()

    