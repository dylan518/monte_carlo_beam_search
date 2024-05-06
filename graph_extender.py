

import torch
from sequence_generator import SequenceGenerator
from graph_manager import BeamSearchGraph
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor


class GraphExtender:
    def __init__(self,bottleneck_size=50,batch_size=128):
        self.sequence_generator = SequenceGenerator()
        self.bottleneck_size = 50  # Number of top probable tokens to consider
        self.batch_size = 128  # Number of nodes to extend in parallel
        self.graph_manager = BeamSearchGraph(top_k=self.bottleneck_size)  # Initialize graph_manager to None        

    def build_graph(self, string):
        """
        Tokenizes the input string and builds the initial graph.

        Args:
            string (str): The input string to tokenize and build the graph from.
        """
        token_tensor = self.sequence_generator.tokenize_string(string)
        tokens = token_tensor[0]  # Convert tensor to list of token IDs
        self.graph_manager = BeamSearchGraph(tokens)

    def process_node(node, top_probs, top_indices):
        new_nodes = OrderedDict()
        for top_prob, top_index in zip(top_probs, top_indices):
            score = node["score"] + math.log(top_prob.item())
            new_node = {
                "depth": node["depth"] + 1,
                "tokens": node["tokens"] + [top_index.item()],
                "score": score
            }
            new_nodes[score] = new_node
        return new_nodes
    


    def extend_graph(self, nodes):
        """
        Extends the graph based on the given nodes by generating and applying the top probable tokens.

        Args:
            nodes (list of dict): List of node dictionaries to extend.
        """
        # Batch the token IDs together
        batched_token_ids = []
        for node in nodes:
            token_ids = node["tokens"]
            batched_token_ids.append(token_ids)

        batched_token_ids = torch.tensor(batched_token_ids, dtype=torch.long)

        # Add batch size dimension if needed
        if batched_token_ids.ndim == 1:
            batched_token_ids = batched_token_ids.unsqueeze(0)

        top_probs_list, top_indices_list = self.sequence_generator.generate_next_token_probs(batched_token_ids, top_n=self.bottleneck_size)

        # Move the results back to CPU for further processing
        top_probs_list = [top_probs.cpu() for top_probs in top_probs_list]
        top_indices_list = [top_indices.cpu() for top_indices in top_indices_list]

        # Use multithreading to process nodes in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for node, top_probs, top_indices in zip(nodes, top_probs_list, top_indices_list):
                future = executor.submit(self.process_node, node, top_probs, top_indices)
                futures.append(future)

            # Gather the results from the futures
            new_nodes = OrderedDict()
            for future in futures:
                node_results = future.result()
                new_nodes.update(node_results)

        self.graph_manager.add_nodes(new_nodes)
                
    
    def run_extension_loop(self, num_iterations):
        """
        Runs the main graph extension loop for a specified number of iterations.

        Args:
            num_iterations (int): Number of iterations to run the extension loop.
        """
        for _ in range(num_iterations):
            # Identify the leaf nodes
            sampled_nodes = self.graph_manager.sample_nodes(self.batch_size)

            # Extend the sampled nodes
            self.extend_graph(sampled_nodes)
    
    def find_highest_prob_leaf_node(self):
        """
        Finds the leaf node with the highest log probability in the graph.

        Returns:
            tuple: A tuple containing the node ID and its log probability.
        """
        return self.graph_manager.best_node["tokens"]
        
    def main(self, string, iterations):
        self.build_graph(string)
        self.run_extension_loop(iterations)
        best_result=self.find_highest_prob_leaf_node()
        self.sequence_generator.token_to_string(best_result)
        return best_result
