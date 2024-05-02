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

    @patch('torch.nn.functional.softmax')
    def test_generate_next_token_probs(self, mock_softmax):
        sequences = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected_probs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.7]], [[0.3, 0.4, 0.3]]])  # Mocked logits
        self.sequence_generator.model.return_value = mock_outputs
        mock_softmax.return_value = expected_probs

        probs = self.sequence_generator.generate_next_token_probs(sequences)

        self.sequence_generator.model.assert_called_once_with(input_ids=sequences)
        mock_softmax.assert_called_once_with(mock_outputs.logits[:, -1, :], dim=-1)
        self.assertTrue(torch.equal(probs, expected_probs))


if __name__ == '__main__':
    unittest.main()