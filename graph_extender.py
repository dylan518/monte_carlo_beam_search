

import torch
from sequence_generator import SequenceGenerator
from graph_manager import BeamSearchGraph
import math
from collections import OrderedDict


class GraphExtender:
    def __init__(self,bottleneck_size=20,batch_size=128,top_k=10):
        self.top_k=5000
        self.sequence_generator = SequenceGenerator()
        self.bottleneck_size = 20  # Number of top probable tokens to consider
        self.batch_size = 512  # Number of nodes to extend in parallel
        self.graph_manager = BeamSearchGraph(top_k=self.top_k)  # Initialize graph_manager to None
        self.count=0

    def build_graph(self, string):
        """
        Tokenizes the input string and builds the initial graph.

        Args:
            string (str): The input string to tokenize and build the graph from.
        """
        token_tensor = self.sequence_generator.tokenize_string(string)
        tokens = token_tensor[0]  # Convert tensor to list of token IDs
        self.graph_manager.build_graph(tokens)


    def process_node(self, node, top_probs, top_indices):
      # Calculate the geometric average of probabilities
      count = node["depth"]
      current_gm = node["score"]
      new_gm = current_gm ** (count / (count + 1)) * top_probs.squeeze() ** (1 / (count + 1))

      # Prepare to concatenate tokens by expanding existing tokens to match top_indices shape
      existing_tokens = node["tokens"].expand(top_probs.size(0), -1)

      # Stack the new tokens with the existing ones
      new_tokens = torch.cat((existing_tokens, top_indices.unsqueeze(1)), dim=1)

      # Create new nodes in a single batch operation
      new_nodes = OrderedDict()
      for i, (gm_score, tokens) in enumerate(zip(new_gm.tolist(), new_tokens)):
          new_node = {
              "depth": node["depth"] + 1,
              "tokens": tokens,
              "score": gm_score
          }
          new_nodes[gm_score] = new_node

      return new_nodes
    

    def process_nodes_batch(self, nodes, top_probs_list, top_indices_list):
        new_nodes = OrderedDict()
        for index, node in enumerate(nodes):
            top_probs = top_probs_list[index]
            top_indices = top_indices_list[index]
            
            # Calculate the geometric average of probabilities
            count = node["depth"]
            current_gm = node["score"]
            new_gm = current_gm ** (count / (count + 1)) * top_probs.squeeze() ** (1 / (count + 1))

            # Move existing_tokens to the CPU
            existing_tokens = node["tokens"].expand(top_probs.size(0), -1).cpu()

            # Stack the new tokens with the existing ones
            new_tokens = torch.cat((existing_tokens, top_indices.unsqueeze(1)), dim=1)

            # Create new nodes
            for i, (gm_score, tokens) in enumerate(zip(new_gm.tolist(), new_tokens)):
                new_node = {
                    "depth": node["depth"] + 1,
                    "tokens": tokens,
                    "score": gm_score
                }
                new_nodes[gm_score] = new_node

        return new_nodes




    def extend_graph(self, nodes):
        """
        Extends the graph based on the given nodes by generating and applying the top probable tokens.

        Args:
            nodes (list of dict): List of node dictionaries to extend.
        """
        start_time = time.time()

        # Batch the token IDs together
        batched_token_ids = []
        if isinstance(nodes, OrderedDict):
            # If nodes is an OrderedDict, iterate over its values
            for node in list(nodes.values()):
                batched_token_ids.append(node["tokens"])
        else:
            # If nodes is a list, iterate over its elements
            for node in nodes:
                token_ids = node["tokens"]
                batched_token_ids.append(token_ids)

        if not batched_token_ids:
            raise ValueError("No nodes to extend.")

        #checkpoint1 = time.time()
        #print(f"Checkpoint 1 (Batching token IDs): {checkpoint1 - start_time:.4f} seconds")
        # Checkpoint 1: Batching token IDs
        # This checkpoint measures the time taken to batch the token IDs together.

        top_probs_list, top_indices_list = self.sequence_generator.generate_next_token_probs(batched_token_ids, top_n=self.bottleneck_size)

        #checkpoint2 = time.time()
        #print(f"Checkpoint 2 (Generating next token probabilities): {checkpoint2 - checkpoint1:.4f} seconds")
        # Checkpoint 2: Generating next token probabilities
        # This checkpoint measures the time taken to generate the top probable tokens and their probabilities.

        # Create a CUDA stream
        stream = torch.cuda.Stream()

        # Allocate memory on the CPU for the results
        top_probs_cpu = torch.empty_like(top_probs_list, device='cpu', pin_memory=True)
        top_indices_cpu = torch.empty_like(top_indices_list, device='cpu', pin_memory=True)

        # Asynchronously move the results to the CPU using the CUDA stream
        with torch.cuda.stream(stream):
            top_probs_cpu.copy_(top_probs_list, non_blocking=True)
            top_indices_cpu.copy_(top_indices_list, non_blocking=True)

        checkpoint3 = time.time()
        #print(f"Checkpoint 3 (Moving results to CPU): {checkpoint3 - checkpoint2:.4f} seconds")
        # Checkpoint 3: Moving results to CPU
        # This checkpoint measures the time taken to move the generated probabilities and indices to the CPU.

        new_nodes = self.process_nodes_batch(nodes, top_probs_cpu, top_indices_cpu)

        # Wait for the data transfer to complete
        stream.synchronize()
        checkpoint4 = time.time()

        #print(f"Checkpoint 4 (Processing nodes): {checkpoint4 - checkpoint3:.4f} seconds")
        # Checkpoint 4: Processing nodes
        # This checkpoint measures the time taken to process each node and generate new nodes.

        self.graph_manager.add_nodes(new_nodes)

        end_time = time.time()
        self.count+=1
        print(f"Total time for extend_graph: {end_time - start_time:.4f} seconds - Print Count: {self.count}", end='\r')

        # Additional checkpoints can be added as needed

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





g=GraphExtender()
string="I love milk"
g.build_graph(string)
#g.graph_manager.print_graph(1)
g.run_extension_loop(100)
g.graph_manager.print_graph(3)
best_result=g.find_highest_prob_leaf_node()

output=g.sequence_generator.token_to_string(best_result)
print(output)
del g


