import unittest
from graph_manager import GraphManager  # Assume your GraphManager class is in a file named graph_manager.py

class TestGraphManager(unittest.TestCase):
    def setUp(self):
        """
        Setup a GraphManager instance with predefined tokens before each test.
        """
        self.tokens = ["I", "am", "a", "programmer"]
        self.manager = GraphManager(self.tokens)

    def test_initialization(self):
        """
        Test initialization of GraphManager with non-empty token list.
        """
        self.assertEqual(len(self.manager.graph.nodes), 4)
        self.assertEqual(self.manager.graph.nodes[1]['token'], "I")




    def test_extend_node(self):
        """
        Test extending a node with vocabulary-probability pairs.
        """
        vocab_probs = {'working': 0.8, 'hard': 0.2}
        self.manager.extend_node(1, vocab_probs)
        # Check that two new nodes have been added
        self.assertEqual(len(self.manager.graph.nodes()), 3 + 2)  # Initial 3 + 2 new ones

        # Check if the new tokens are in the graph
        found_tokens = {self.manager.graph.nodes[node_id]['token'] for node_id in self.manager.graph.nodes()}
        self.assertIn('working', found_tokens)
        self.assertIn('hard', found_tokens)

    def test_batch_extend_graph(self):
        """
        Test batch extending the graph.
        """
        nodes_data = [
            (1, {'developer': 0.6, 'artist': 0.4})
        ]
        self.manager.batch_extend_graph(nodes_data)
        # Check if the new tokens are in the graph
        found_tokens = {self.manager.graph.nodes[node_id]['token'] for node_id in self.manager.graph.nodes()}
        self.assertIn('developer', found_tokens)
        self.assertIn('artist', found_tokens)


    def test_identify_leaf_nodes(self):
        """
        Test identification of leaf nodes.
        """
        leaf_nodes = self.manager.identify_leaf_nodes()
        self.assertIn(4, leaf_nodes)  # Assuming the last token node is a leaf

    def test_sample_leaf_nodes(self):
        """
        Test sampling of leaf nodes.
        """
        sampled_nodes = self.manager.sample_leaf_nodes(1)
        self.assertTrue(sampled_nodes[0] in [1, 2, 3, 4])

    def test_reconstruct_sentence(self):
        """
        Test reconstructing a sentence from a specific node.
        """
        sentence = self.manager.reconstruct_sentence(4)
        self.assertEqual(sentence, "I am a programmer")

    def test_find_highest_prob_leaf_node(self):
        """
        Test finding the highest probability leaf node.
        """
        self.manager.graph.nodes[1]['score'] = -1.0
        self.manager.graph.nodes[2]['score'] = -0.5
        self.manager.graph.nodes[3]['score'] = -0.2
        self.manager.graph.nodes[4]['score'] = -0.1  # Highest score
        node_id, log_prob = self.manager.find_highest_prob_leaf_node()
        self.assertEqual(node_id, 4)
        self.assertEqual(log_prob, -0.1)

if __name__ == '__main__':
    unittest.main()
