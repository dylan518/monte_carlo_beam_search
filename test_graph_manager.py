import unittest
from graph_manager import GraphManager  # Ensure your GraphManager is correctly imported

class TestGraphManager(unittest.TestCase):
    def setUp(self):
        self.tokens = ["I", "am", "a", "programmer"]
        self.manager = GraphManager(self.tokens)

    def test_extend_node(self):
        """
        Test extending a node and check the correct tokens and number of nodes are added.
        """
        node_id = 4  # Extending the last initial node
        vocab_probs = {'working': 0.8, 'hard': 0.2}
        self.manager.extend_node(node_id, vocab_probs)
        self.assertEqual(len(self.manager.graph.nodes), 6)  # Initial 4 + 2 new ones

        # Collect all tokens in the graph to ensure 'working' and 'hard' are added
        tokens_in_graph = [self.manager.graph.nodes[node]['token'] for node in self.manager.graph.nodes()]
        self.assertIn('working', tokens_in_graph)
        self.assertIn('hard', tokens_in_graph)

    def test_batch_extend_graph(self):
        """
        Test batch extending the graph and verify the correct tokens are added.
        """
        nodes_data = [
            (4, {'developer': 0.6, 'artist': 0.4})  # Extending node 4
        ]
        self.manager.batch_extend_graph(nodes_data)
        
        # Collect all tokens in the graph to ensure 'developer' and 'artist' are added
        tokens_in_graph = [self.manager.graph.nodes[node]['token'] for node in self.manager.graph.nodes()]
        self.assertIn('developer', tokens_in_graph)
        self.assertIn('artist', tokens_in_graph)

    def test_identify_leaf_nodes(self):
        """
        Test identifying leaf nodes after extending the graph.
        """
        self.manager.extend_node(4, {'new_token': 0.5})  # Extend node 4 to create a new leaf
        leaf_nodes = self.manager.identify_leaf_nodes()
        self.assertIn(5, leaf_nodes)  # New leaf node ID should be 5

    def test_reconstruct_sentence(self):
        """
        Test reconstructing the sentence from the extended node.
        """
        self.manager.extend_node(4, {'new_token': 0.5})
        sentence = self.manager.reconstruct_sentence(5)  # New node ID 5
        self.assertEqual(sentence, "I am a programmer new_token")

    def test_find_highest_prob_leaf_node(self):
        """
        Ensure the correct leaf node is identified as having the highest log probability.
        """
        self.manager.extend_node(4, {'new_token': 0.5})  # Extend node 4
        self.manager.graph.nodes[5]['score'] = -0.1  # Set a high score
        node_id, log_prob = self.manager.find_highest_prob_leaf_node()
        self.assertEqual(len(self.manager.graph.nodes), 5)
        self.assertEqual(node_id, 5)
        self.assertEqual(log_prob, -0.1)

if __name__ == '__main__':
    unittest.main()
