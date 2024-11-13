import unittest
from sequence_generator import SequenceGenerator
from graph_manager import GraphManager
from graph_extender import GraphExtender

class TestGraphExtender(unittest.TestCase):
    def setUp(self):
        self.extender = GraphExtender()
        self.extender.sequence_generator = SequenceGenerator()
        self.input_string = "This is a test string."

    def test_graph_extension(self):
        # Build the initial graph
        self.extender.build_graph(self.input_string)

        # Check if the graph_manager is initialized correctly
        self.assertIsInstance(self.extender.graph_manager, GraphManager)
        initial_nodes = list(self.extender.graph_manager.graph.nodes(data='token'))
        self.assertEqual(len(initial_nodes), len(self.input_string.split()))

        # Run the extension loop for a specified number of iterations
        num_iterations = 3
        self.extender.run_extension_loop(num_iterations)

        # Check if the graph has been extended
        extended_nodes = list(self.extender.graph_manager.graph.nodes(data='token'))
        self.assertGreater(len(extended_nodes), len(initial_nodes))

        # Check if the leaf nodes have been updated
        leaf_nodes = self.extender.graph_manager.identify_leaf_nodes()
        self.assertGreater(len(leaf_nodes), 0)

        # Find the highest probability leaf node
        highest_prob_node, highest_prob = self.extender.find_highest_prob_leaf_node()
        self.assertIsNotNone(highest_prob_node)
        self.assertIsInstance(highest_prob, float)

        # Reconstruct the sentence from the highest probability leaf node
        reconstructed_sentence = self.extender.graph_manager.reconstruct_sentence(highest_prob_node)
        self.assertIsInstance(reconstructed_sentence, str)
        self.assertGreater(len(reconstructed_sentence), len(self.input_string))

if __name__ == '__main__':
    unittest.main()