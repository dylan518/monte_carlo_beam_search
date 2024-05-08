from collections import OrderedDict
import numpy as np


class BeamSearchGraph:
    def __init__(self, top_k):
        self.leaf_nodes = OrderedDict()
        self.best_node = None
        self.top_k = top_k

    def build_graph(self, initial_tokens, initial_score=0.0001):
        initial_node=OrderedDict()
        initial_node[initial_score] = {
            'depth': 0,
            'tokens': initial_tokens,
            'score': initial_score
        }
        self.leaf_nodes.update(initial_node)
        self.best_node=initial_node[initial_score]

    def add_nodes(self, nodes):
        # Update the best node if necessary
        best_score = max(nodes.keys())
        if self.best_node is None or best_score > self.best_node["score"]:
            self.best_node = nodes[best_score]

        # Merge the new nodes with the existing leaf_nodes dictionary
        self.leaf_nodes.update(nodes)


        # Keep only the top_k nodes
        if len(self.leaf_nodes) > self.top_k:
            # Remove the lowest scoring nodes
            for _ in range(len(self.leaf_nodes) - self.top_k):
                self.leaf_nodes.popitem(last=True)

    def sample_nodes(self, n_samples):
        if not self.leaf_nodes:
            return []

        # Extract scores and convert from log probability to probability
        scores = np.array(list(self.leaf_nodes.keys()))
        if n_samples>=len(scores):
          samples=list(self.leaf_nodes.values())
          self.leaf_nodes.clear()
          return samples


        # Normalize the scores
        normalized_score = scores / np.sum(scores)


        # Perform weighted sampling
        sampled_indices = np.random.choice(len(normalized_score), size=n_samples, replace=False, p=normalized_score)

        # Retrieve the sampled nodes and remove them from the leaf_nodes dictionary
        sampled_nodes = []
        for index in sampled_indices:
            score = scores[index]
            node = self.leaf_nodes.pop(score)
            sampled_nodes.append(node)

        return sampled_nodes
    def print_graph(self, n_nodes):
        print(f"Best node: {self.best_node}")
        print(f"Top {n_nodes} leaf nodes:")

        for i, (score, node) in enumerate(self.leaf_nodes.items(), start=1):
            print(f"Node {i}:")
            print(f"  Score: {score}")
            print(f"  Depth: {node['depth']}")
            print(f"  Tokens: {node['tokens']}")

            if i == n_nodes:
                break
