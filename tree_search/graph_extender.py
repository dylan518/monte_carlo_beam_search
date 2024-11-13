# graph_extender.py

import torch
import numpy as np
from beam_search_graph import BeamSearchGraph

class GraphExtender:
    def __init__(self, sequence_generator, bottleneck_size=20, batch_size=16, max_leaf_nodes=1000):
        self.sequence_generator = sequence_generator
        self.bottleneck_size = bottleneck_size
        self.batch_size = batch_size  # Number of nodes to expand in each iteration
        self.max_leaf_nodes = max_leaf_nodes  # Maximum number of leaf nodes to keep
        self.graph_manager = BeamSearchGraph()
        self.count = 0

        # Compute the global average once during initialization
        self.global_avg = self.sequence_generator.compute_mean_confidence()
        print(f"Global average (mean model output confidence): {self.global_avg}")

        # Get the EOS token ID
        self.eos_token_id = self.sequence_generator.tokenizer.eos_token_id
        if self.eos_token_id is None:
            print("Warning: eos_token_id is None")
        else:
            print(f"EOS token ID: {self.eos_token_id}")

    def clear_memory(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def build_graph(self, string):
        token_tensor = self.sequence_generator.tokenize_string(string)
        tokens = token_tensor[0].tolist()  # Convert to list
        initial_node = {
            'depth': 0,
            'tokens': tokens,
            'sum_x': [],
            'score': 0.0,  # Initial score
            'is_completed': False  # New field
        }
        self.graph_manager.leaf_nodes.append(initial_node)

    def process_nodes_batch(self, nodes, top_probs_list, top_indices_list):
        new_nodes = []
        for index, node in enumerate(nodes):
            adjusted_probs = top_probs_list[index]  # Tensor of shape [bottleneck_size]
            top_indices = top_indices_list[index]    # Tensor of shape [bottleneck_size]

            existing_sum_x = node.get("sum_x", [])

            existing_tokens = node["tokens"]

            # Expand existing tokens to match the number of top tokens
            existing_tokens_expanded = [existing_tokens] * len(top_indices)

            # Concatenate each top_index to the existing tokens
            new_tokens = [existing_tokens_expanded[i] + [top_indices[i].item()] for i in range(len(top_indices))]

            adjusted_probs_np = adjusted_probs.cpu().numpy()  # Shape: [bottleneck_size]

            for i in range(len(adjusted_probs_np)):
                adjusted_prob = adjusted_probs_np[i]
                tokens = new_tokens[i]

                # Check if the last token is eos_token_id
                is_completed = (tokens[-1] == self.eos_token_id)

                # Update sum_x
                new_sum_x = existing_sum_x + [adjusted_prob]

                # Compute score using the mapping
                score = np.mean(new_sum_x)

                # Create new node
                new_node = {
                    "depth": node["depth"] + 1,
                    "tokens": tokens,
                    "sum_x": new_sum_x,  # List of adjusted_probs
                    'score': score,
                    'is_completed': is_completed  # New field
                }

                new_nodes.append(new_node)
                self.count += 1

        return new_nodes

    def extend_graph(self, nodes):
        batched_token_ids = []
        for node in nodes:
            token_ids = torch.tensor(node["tokens"], dtype=torch.long)
            batched_token_ids.append(token_ids)

        if not batched_token_ids:
            return []

        top_probs_list, top_indices_list = self.sequence_generator.generate_next_token_probs(
            batched_token_ids, top_n=self.bottleneck_size)

        new_nodes = self.process_nodes_batch(nodes, top_probs_list, top_indices_list)

        return new_nodes

    def run_extension_loop(self, num_iterations):
        for i in range(num_iterations):
            # Get nodes to expand
            nodes_to_expand = self.graph_manager.get_nodes_to_expand(self.batch_size)

            if not nodes_to_expand:
                print("No nodes to expand (all nodes are completed). Exiting loop.")
                break

            # Remove nodes to expand from leaf_nodes
            self.graph_manager.remove_nodes(nodes_to_expand)

            # Extend nodes
            new_nodes = self.extend_graph(nodes_to_expand)

            if not new_nodes:
                print("No new nodes generated. Exiting loop.")
                break

            # Add new nodes to leaf_nodes
            self.graph_manager.add_nodes(new_nodes)

            # Keep the top max_leaf_nodes to limit memory usage
            self.graph_manager.filter_nodes(self.max_leaf_nodes)

            if i % 10 == 0:
                self.clear_memory()

        # After iterations, we might want to clear memory one last time
        self.clear_memory()

    def find_best_node(self):
        best_tokens = self.graph_manager.find_best_node()
        if best_tokens is None:
            return None
        result_string = self.sequence_generator.token_to_string(best_tokens)
        return result_string

    def main(self, string, iterations):
        self.build_graph(string)
        self.run_extension_loop(iterations)
        best_result = self.find_best_node()
        if best_result is None:
            return None
        else:
            print(f"Best result: {best_result}")
            return best_result
