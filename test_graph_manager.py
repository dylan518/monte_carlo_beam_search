import unittest
from unittest.mock import patch, MagicMock
from graph_manager import GraphManager


class TestGraphManager(unittest.TestCase):
    def setUp(self):
        self.tokens = ["I", "am", "a", "programmer"]
        self.graph_manager = GraphManager(self.tokens)

    def test_init_with_valid_tokens(self):
        self.assertIsInstance(self.graph_manager.graph, nx.DiGraph)
        self.assertEqual(self.graph_manager.graph.number_of_nodes(), len(self.tokens))
        self.assertEqual(self.graph_manager.graph.number_of_edges(), len(self.tokens) - 1)

    def test_init_with_empty_tokens(self):
        with self.assertRaises(ValueError):
            GraphManager([])

    def test_extend_node(self):
        node_id = 1
        vocab_probs = {5: 0.2, 6: 0.8}
        new_nodes, new_edges = self.graph_manager.extend_node(node_id, vocab_probs)
        self.assertEqual(len(new_nodes), len(vocab_probs))
        self.assertEqual(len(new_edges), len(vocab_probs))

    @patch('graph_manager.ThreadPoolExecutor')
    def test_batch_extend_graph(self, mock_executor):
        nodes_data = [(1, {5: 0.2, 6: 0.8}), (2, {7: 0.5, 8: 0.5})]
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [MagicMock(), MagicMock()]

        self.graph_manager.batch_extend_graph(nodes_data)

        self.assertEqual(mock_executor.return_value.__enter__.return_value.submit.call_count, len(nodes_data))

    @patch('graph_manager.ThreadPoolExecutor')
    def test_identify_leaf_nodes(self, mock_executor):
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [MagicMock(return_value=True),
                                                                                MagicMock(return_value=False),
                                                                                MagicMock(return_value=True),
                                                                                MagicMock(return_value=True)]

        leaf_nodes = self.graph_manager.identify_leaf_nodes()

        self.assertEqual(len(leaf_nodes), 3)

    @patch('graph_manager.ThreadPoolExecutor')
    @patch('graph_manager.np.random.choice')
    def test_sample_leaf_nodes(self, mock_choice, mock_executor):
        leaf_nodes = [3, 4]
        num_samples = 2
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [MagicMock(return_value=-1.0),
                                                                                MagicMock(return_value=-0.5)]
        mock_choice.return_value = leaf_nodes

        sampled_nodes = self.graph_manager.sample_leaf_nodes(num_samples)

        self.assertEqual(sampled_nodes, leaf_nodes)

    def test_reconstruct_sentence(self):
        end_node_id = 4
        expected_sentence = "I am a programmer"

        reconstructed_sentence = self.graph_manager.reconstruct_sentence(end_node_id)

        self.assertEqual(reconstructed_sentence, expected_sentence)

    def test_find_highest_prob_leaf_node(self):
        self.graph_manager.graph.nodes[3]['score'] = -1.0
        self.graph_manager.graph.nodes[4]['score'] = -0.5

        max_node, max_score = self.graph_manager.find_highest_prob_leaf_node()

        self.assertEqual(max_node, 4)
        self.assertAlmostEqual(max_score, -0.5)


if __name__ == '__main__':
    unittest.main()