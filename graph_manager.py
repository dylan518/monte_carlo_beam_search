
import networkx as nx
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


class GraphManager:
    def __init__(self, tokens):
        self.graph = nx.DiGraph()
        self.leaf_nodes = {}
        self.leaf_parent_cache = {}
        self.original_leaf_nodes = set()  # Added set to track original leaf nodes
        if not tokens:
            raise ValueError("Tokens list cannot be empty.")
        self.current_id = 0
        self._build_graph(tokens)
        

    def _build_graph(self, tokens):
        prev_node_id = None
        for i, token in enumerate(tokens):
            node_id = self.current_id
            self.current_id += 1
            self.graph.add_node(node_id, token=token, score=0.0001, prev_node=prev_node_id)
            if prev_node_id is not None:
                self.graph.add_edge(prev_node_id, node_id)
            prev_node_id = node_id
            self.leaf_nodes[node_id] = self.graph.nodes[node_id]
        if len(tokens) > 1:
            self.leaf_nodes.pop(1, None)
        self.original_leaf_nodes = set(self.leaf_nodes.keys())

    def extend_node(self, node_id, vocab_probs):
        current_score = self.graph.nodes[node_id]['score']
        if node_id in self.leaf_nodes:
            del self.leaf_nodes[node_id]
            if node_id not in self.original_leaf_nodes:  # Check if the node is not an original leaf node
                if node_id in self.leaf_parent_cache:
                    del self.leaf_parent_cache[node_id]
        new_nodes = {}
        for token, prob in vocab_probs.items():
            new_score = current_score + math.log(prob)
            new_node_id = self.current_id
            self.current_id += 1
            new_node_data = {'token': token, 'score': new_score, 'prev_node': node_id}
            self.graph.add_node(new_node_id, **new_node_data)
            self.graph.add_edge(node_id, new_node_id)
            self.leaf_nodes[new_node_id] = new_node_data
            if node_id not in self.original_leaf_nodes:  # Check if the node is not an original leaf node
                self.leaf_parent_cache[new_node_id] = (node_id, token)

    def reconstruct_sentence(self, end_node_id):
        sequence = []
        current_node_id = end_node_id
        while current_node_id is not None:
            if current_node_id in self.leaf_parent_cache:
                parent_id, token = self.leaf_parent_cache[current_node_id]
                sequence.append(token)
                current_node_id = parent_id
            else:
                node = self.graph.nodes[current_node_id]
                token = node['token']
                if token.startswith('Ġ'):
                    token = ' ' + token[1:]
                sequence.append(token)
                current_node_id = node.get('prev_node')

        sentence = ''.join(sequence[::-1])
        return sentence

    def identify_leaf_nodes(self):
        return list(self.leaf_nodes.keys())  # Return the current set of leaf node IDs

    def sample_leaf_nodes(self, num_samples):
        if not self.leaf_nodes:
            return []

        leaf_node_ids = list(self.leaf_nodes.keys())
        scores = [node['score'] for node in self.leaf_nodes.values()]

        # Convert log scores to probabilities
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()  # Normalize to form a valid probability distribution

        # Perform weighted sampling without replacement
        sampled_nodes = np.random.choice(leaf_node_ids, size=min(num_samples, len(leaf_node_ids)), replace=False, p=probabilities)
        return sampled_nodes.tolist()
    
    def reconstruct_sentence(self, end_node_id):
        sequence = []
        current_node_id = end_node_id
        while current_node_id is not None:
            if current_node_id in self.leaf_parent_cache:
                parent_id, token = self.leaf_parent_cache[current_node_id]
                sequence.append(token)
                current_node_id = parent_id
            else:
                node = self.graph.nodes[current_node_id]
                token = node['token']
                if token.startswith('Ġ'):
                    token = ' ' + token[1:]
                sequence.append(token)
                current_node_id = node.get('prev_node')

        sentence = ''.join(sequence[::-1])
        return sentence
    
    def find_highest_prob_leaf_node(self):
        """
        Finds the leaf node with the highest log probability in the graph.

        Returns:
            tuple: A tuple containing the node ID and its log probability.
        """
        max_node = None
        max_score = float('-inf')  # Start with the lowest possible log probability

        # Check if a node is a leaf node (no outgoing edges)
        leaf_nodes = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

        for node_id in leaf_nodes:
            node_score = self.graph.nodes[node_id]['score']
            if node_score > max_score:
                max_score = node_score
                max_node = node_id

        return (max_node, max_score) if max_node is not None else (None, None)

    def get_node_content(self, node_id):
        """
        Retrieve the content (tokens or sentence) associated with a node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            str: The content of the node.
        """
        # Assuming each node has a 'content' attribute or similar
        return {"score":self.graph.nodes[node_id]['score'],"token":self.graph.nodes[node_id]['token']}

    def print_graph(self):
        """
        Print all nodes in the graph with their content.
        """
        for node_id in self.graph.nodes:
            content = self.get_node_content(node_id)
            print(f"Node {node_id}: {content}")





# Example usage assuming necessary classes and graph initialization
if __name__ == "__main__":
    graph=GraphManager(["I", "am", "a", "programmer"])



