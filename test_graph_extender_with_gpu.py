import unittest
from unittest.mock import MagicMock, patch
from graph_extender import GraphExtender
from graph_manager import GraphManager

class TestGraphExtender(unittest.TestCase):
    def setUp(self):
        self.extender = GraphExtender()
        self.input_string = "This is a test string."

    def test_build_graph(self):
        # Mock the tokenize_string method to return a fixed tensor
        mock_tokenize = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        self.extender.sequence_generator.tokenize_string = mock_tokenize

        # Call the build_graph method
        self.extender.build_graph(self.input_string)

        # Check if the tokenize_string method was called with the correct argument
        mock_tokenize.assert_called_once_with(self.input_string)

        # Check if the graph_manager is initialized correctly
        self.assertIsInstance(self.extender.graph_manager, GraphManager)
        self.assertListEqual(list(self.extender.graph_manager.graph.nodes(data='token')),
                             [(1, {'token': 1}), (2, {'token': 2}), (3, {'token': 3}), (4, {'token': 4}), (5, {'token': 5})])

    @patch('graph_manager.GraphManager.extend_node')
    def test_extend_graph(self, mock_extend_node):
        # Create a mock graph with nodes
        self.extender.graph_manager = MagicMock()
        self.extender.graph_manager.graph.nodes = {
            1: {'token': 'A'},
            2: {'token': 'B'},
            3: {'token': 'C'}
        }

        # Mock the generate_next_token_probs method to return fixed probabilities
        mock_probs = {
            1: {'D': 0.4, 'E': 0.3, 'F': 0.2, 'G': 0.1},
            2: {'H': 0.5, 'I': 0.3, 'J': 0.2},
            3: {'K': 0.6, 'L': 0.4}
        }
        self.extender.sequence_generator.generate_next_token_probs = MagicMock(return_value=mock_probs)

        # Call the extend_graph method
        node_ids = [1, 2, 3]
        self.extender.extend_graph(node_ids)

        # Check if the extend_node method was called with the correct arguments
        expected_calls = [
            ((1, {'D': 0.4, 'E': 0.3}), {}),
            ((2, {'H': 0.5, 'I': 0.3}), {}),
            ((3, {'K': 0.6, 'L': 0.4}), {})
        ]
        self.assertListEqual(mock_extend_node.call_args_list, expected_calls)

    # Add more test methods for other methods in the GraphExtender class

if __name__ == '__main__':
    unittest.main()