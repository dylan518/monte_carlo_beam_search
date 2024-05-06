from collections import OrderedDict
import numpy as np

class BeamSearchGraph:
    def __init__(self, top_k):
        self.leaf_nodes = OrderedDict()
        self.best_node = None
        self.top_k = top_k

    def build_graph(self, initial_tokens, initial_score=-1):
        initial_node = {
            'depth': 0,
            'tokens': initial_tokens,
            'score': initial_score
        }
        if not initial_node['tokens']:
            raise ValueError("Initial tokens must not be empty.")
        self.add_nodes([initial_node])
    
    def add_nodes(self, nodes):
        # Sort the input nodes by score in descending order
        sorted_nodes = sorted(nodes, key=lambda x: x['score'], reverse=True)

        # Create a new OrderedDict from the sorted nodes
        new_nodes = OrderedDict((node['score'], node) for node in sorted_nodes)

        # Update the best node if necessary
        if self.best_node is None or sorted_nodes[0]['score'] > self.best_node['score']:
            self.best_node = sorted_nodes[0]

        # Merge the new nodes with the existing leaf_nodes dictionary
        self.leaf_nodes.update(new_nodes)

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
        probabilities = np.exp(scores)



        # Perform weighted sampling
        sampled_indices = np.random.choice(len(scores), size=n_samples, replace=False, p=probabilities)

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