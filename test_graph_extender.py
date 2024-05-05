import unittest
from unittest.mock import MagicMock, patch
from graph_extender import GraphExtender
from graph_manager import GraphManager
from unittest.mock import call
import torch

class TestGraphExtender(unittest.TestCase):
    def setUp(self):
        with patch('sequence_generator.SequenceGenerator') as mock_sequence_generator:
            self.extender = GraphExtender()
            self.extender.sequence_generator = mock_sequence_generator
        self.input_string = "This is a test string."

    def test_build_graph(self):
        # Mock the tokenize_string method to return a fixed tensor
        mock_tokenize = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        self.extender.sequence_generator.tokenize_string.return_value = mock_tokenize.return_value

        # Call the build_graph method
        self.extender.build_graph(self.input_string)

        # Check if the tokenize_string method was called with the correct argument
        self.extender.sequence_generator.tokenize_string.assert_called_once_with(self.input_string)

        # Check if the graph_manager is initialized correctly
        self.assertIsInstance(self.extender.graph_manager, GraphManager)
        self.assertListEqual(list(self.extender.graph_manager.graph.nodes(data='token')),
                             [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    @patch('graph_manager.GraphManager.extend_node')
    def test_extend_graph(self, mock_extend_node):
        # Create an actual graph with nodes
        graph_manager = GraphManager([10, 20, 30])
        self.extender.graph_manager = graph_manager

        # Mock the generate_next_token_probs method to return fixed probabilities
        mock_probs = {
            10: {40: 0.4, 50: 0.3, 60: 0.2, 70: 0.1},
            20: {80: 0.5, 90: 0.3, 100: 0.2},
            30: {110: 0.6, 120: 0.4}
        }
        self.extender.sequence_generator.generate_next_token_probs.return_value = mock_probs

        # Call the extend_graph method
        node_ids = [1, 2, 3]
        self.extender.extend_graph(node_ids)

        # Check if the extend_node method was called with the correct arguments
        expected_calls = [
            call(1, {40: 0.4, 50: 0.3, 60: 0.2, 70: 0.1}),
            call(2, {80: 0.5, 90: 0.3, 100: 0.2}),
            call(3, {110: 0.6, 120: 0.4})
        ]
        self.assertEqual(mock_extend_node.call_count, len(expected_calls))
        mock_extend_node.assert_has_calls(expected_calls, any_order=True)
if __name__ == '__main__':
    unittest.main()