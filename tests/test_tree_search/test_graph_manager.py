import unittest
import numpy as np
from collections import OrderedDict
from graph_manager import BeamSearchGraph
from collections import OrderedDict


class TestBeamSearchGraph(unittest.TestCase):
    def setUp(self):
        self.top_k = 3
        self.graph = BeamSearchGraph(top_k=self.top_k)

    def test_build_graph(self):
        initial_tokens = ['The', 'quick', 'brown', 'fox']
        initial_score = -10.0
        self.graph.build_graph(initial_tokens, initial_score)

        self.assertEqual(len(self.graph.leaf_nodes), 1)
        self.assertIsNotNone(self.graph.best_node)
        self.assertEqual(self.graph.best_node['score'], initial_score)
        self.assertEqual(self.graph.best_node['tokens'], initial_tokens)

    def test_add_nodes(self):
        nodes = [
            {'depth': 1, 'tokens': ['The', 'quick'], 'score': -5.0},
            {'depth': 1, 'tokens': ['The', 'swift'], 'score': -7.0},
            {'depth': 1, 'tokens': ['The', 'fast'], 'score': -6.0},
            {'depth': 1, 'tokens': ['The', 'slow'], 'score': -8.0}
        ]
        self.graph.add_nodes(nodes)

        self.assertEqual(len(self.graph.leaf_nodes), self.top_k)
        self.assertEqual(self.graph.best_node['score'], -5.0)
        self.assertEqual(self.graph.best_node['tokens'], ['The', 'quick'])

    def test_sample_nodes(self):
        nodes = [
            {'depth': 1, 'tokens': ['The', 'quick'], 'score': -5.0},
            {'depth': 1, 'tokens': ['The', 'swift'], 'score': -7.0},
            {'depth': 1, 'tokens': ['The', 'fast'], 'score': -6.0},
            {'depth': 1, 'tokens': ['The', 'slow'], 'score': -8.0}
        ]
        self.graph.add_nodes(nodes)

        n_samples = 2
        sampled_nodes = self.graph.sample_nodes(n_samples)

        self.assertEqual(len(sampled_nodes), n_samples)
        self.assertEqual(len(self.graph.leaf_nodes), self.top_k - n_samples)

    def test_print_graph(self):
        nodes = [
            {'depth': 1, 'tokens': ['The', 'quick'], 'score': -5.0},
            {'depth': 1, 'tokens': ['The', 'swift'], 'score': -7.0},
            {'depth': 1, 'tokens': ['The', 'fast'], 'score': -6.0},
            {'depth': 1, 'tokens': ['The', 'slow'], 'score': -8.0}
        ]
        self.graph.add_nodes(nodes)

        n_nodes = 2
        # Redirect stdout to a buffer
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.graph.print_graph(n_nodes)

        # Check the printed output
        output = buffer.getvalue()
        self.assertIn("Best node:", output)
        self.assertIn("Top 2 leaf nodes:", output)
        self.assertIn("Node 1:", output)
        self.assertIn("Node 2:", output)

if __name__ == '__main__':
    unittest.main()