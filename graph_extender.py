import torch
from sequence_generator import SequenceGenerator
from graph_manager import GraphManager

class GraphExtender:
    def __init__(self,bottleneck_size=50,batch_size=128):
        self.sequence_generator = SequenceGenerator()
        self.bottleneck_size = 50  # Number of top probable tokens to consider
        self.batch_size = 128  # Number of nodes to extend in parallel
        self.graph_manager = None  # Initialize graph_manager to None

    def build_graph(self, string):
        """
        Tokenizes the input string and builds the initial graph.

        Args:
            string (str): The input string to tokenize and build the graph from.
        """
        tokens_tensor = self.sequence_generator.tokenize_string(string)
        tokens=self.sequence_generator.decode_token_tensor(tokens_tensor)
        tokens = tokens[0]  # Convert tensor to list of token IDs
        self.graph_manager = GraphManager(tokens)

    def extend_graph(self, node_ids):
        """
        Extends the graph based on the given node IDs by generating and applying the top probable tokens.

        Args:
            node_ids (list of int): List of node IDs to extend.
        """
        sentences = []
        for node_id in node_ids:
            sentence = self.graph_manager.reconstruct_sentence(node_id)
            sentences.append(sentence)

        tokenized_sentences = self.sequence_generator.tokenize_string(sentences)
        top_probs, top_indices = self.sequence_generator.generate_next_token_probs(tokenized_sentences, top_n=self.bottleneck_size)

        # Process each node
        for i, node_id in enumerate(node_ids):
            # Decode tokens using the method in SequenceGenerator
            tokens = self.sequence_generator.decode_token_tensor(top_indices[i])
            probabilities = top_probs[i].tolist()

            # Create a dictionary of token-probability pairs
            token_prob_dict = {tokens[j]: probabilities[j] for j in range(len(tokens))}

            # Extend the node with the selected vocab probabilities
            self.graph_manager.extend_node(node_id, token_prob_dict)
    
    def run_extension_loop(self, num_iterations):
        """
        Runs the main graph extension loop for a specified number of iterations.

        Args:
            num_iterations (int): Number of iterations to run the extension loop.
        """
        for _ in range(num_iterations):
            # Identify the leaf nodes
            leaf_nodes = self.graph_manager.identify_leaf_nodes()

            # Sample a subset of leaf nodes
            num_samples = min(len(leaf_nodes), self.batch_size)
            sampled_nodes = self.graph_manager.sample_leaf_nodes(num_samples)

            # Extend the sampled nodes
            self.extend_graph(sampled_nodes)
    
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
        
    def main(self, string, iterations):
        self.build_graph(string)
        self.run_extension_loop(iterations)
        best_result=self.graph_manager.find_highest_prob_leaf_node()
        self.graph_manager.reconstruct_sentence(best_result[0])
        return best_result
