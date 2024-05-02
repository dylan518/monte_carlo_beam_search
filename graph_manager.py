import networkx as nx
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

class GraphManager:
    def __init__(self, tokens):
        """
        Initializes the GraphManager with a list of tokens and creates a directed graph.

        Args:
            tokens (list of str): List of tokens to        """
        self.graph = nx.DiGraph()
        if not tokens:
            raise ValueError("Tokens list cannot be empty.")
        self._build_graph(tokens)
        
    def _build_graph(self, tokens):
        """
        Builds the initial graph structure based on the given list of tokens.

        Args:
            tokens (list of str): List of tokens to build the graph.
        """
        for i in range(len(tokens)):
            node_id = i + 1
            token = tokens[i]
            self.graph.add_node(node_id, token=token, score=99, prev_node=None)

            if i > 0:
                prev_node_id = i
                self.graph.add_edge(prev_node_id, node_id)
                self.graph.nodes[node_id]['prev_node'] = prev_node_id

    def extend_node(self, node_id, vocab_probs):
        """
        Extends a graph node with given vocabulary-probability pairs.

        Args:
            node_id (int): The ID of the node being extended.
            vocab_probs (dict): A dictionary of next possible tokens and their probabilities.
        """
        current_score = self.graph.nodes[node_id]['score']
        new_nodes = {}
        new_edges = []
        for token, prob in vocab_probs.items():
            new_score = current_score + math.log(prob)  # Update score using log probability
            new_node_id = max(self.graph.nodes, default=0) + 1  # Ensure unique node ID
            new_nodes[new_node_id] = {'token': token, 'score': new_score, 'prev_node': node_id}
            new_edges.append((node_id, new_node_id, prob))
        return new_nodes, new_edges


    
    def batch_extend_graph(self, nodes_data):
        """
        Batch extends the graph using a list of node data, each with vocab-probability pairs.

        Args:
            nodes_data (list of tuples): Each tuple contains (node_id, vocab_probs).
        """
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.extend_node, node_id, vocab_probs) 
                       for node_id, vocab_probs in nodes_data]

            for future in as_completed(futures):
                new_nodes, new_edges = future.result()
                for node_id, node_data in new_nodes.items():
                    self.graph.add_node(node_id, **node_data)
                for src_id, dest_id, weight in new_edges:
                    self.graph.add_edge(src_id, dest_id, weight=weight)
    
    def identify_leaf_nodes(self):
        """
        Identifies all leaf nodes in the graph (nodes with no outgoing edges).
        
        Returns:
            list: List of leaf node IDs.
        """
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda node: self.graph.out_degree(node) == 0, node)
                       for node in self.graph.nodes()]
            leaf_nodes = [node for node, is_leaf in zip(self.graph.nodes(), as_completed(futures)) if is_leaf.result()]
        return leaf_nodes

    def sample_leaf_nodes(self, num_samples):
        """
        Samples leaf nodes based on their scores converted from log probabilities to probabilities.
        
        Args:
            num_samples (int): The number of nodes to sample.
        
        Returns:
            list: List of sampled leaf node IDs.
        """
        leaf_nodes = self.identify_leaf_nodes()
        if not leaf_nodes:
            return []

        with ThreadPoolExecutor() as executor:
            future_scores = {executor.submit(lambda node: self.graph.nodes[node]['score'], node): node
                             for node in leaf_nodes}
            scores = [future.result() for future in as_completed(future_scores)]

        # Convert log scores to probabilities
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()  # Normalize to form a valid probability distribution

        # Perform weighted sampling without replacement
        sampled_nodes = np.random.choice(leaf_nodes, size=min(num_samples, len(leaf_nodes)), replace=False, p=probabilities)
        return sampled_nodes.tolist()
    
    def reconstruct_sentence(self, end_node_id):
        """
        Reconstructs a sentence by tracing back from the specified end node to the original parent node in the DAG.
        
        Args:
            end_node_id (int): The ID of the end node from which to start tracing back.
        
        Returns:
            str: The reconstructed sentence from the graph.
        """
        sequence = []
        current_node_id = end_node_id
        while current_node_id is not None:
            node = self.graph.nodes[current_node_id]
            sequence.append(node['token'])
            current_node_id = node.get('prev_node')  # Get the previous node

        # Reverse the sequence to start from the root and convert tokens back to string
        return ' '.join(map(str, sequence[::-1]))
    
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





# Example usage assuming necessary classes and graph initialization
if __name__ == "__main__":
    graph=GraphManager(["I", "am", "a", "programmer"])

