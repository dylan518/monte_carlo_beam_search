# beam_search_graph.py

class BeamSearchGraph:
    def __init__(self):
        self.leaf_nodes = []

    def add_nodes(self, nodes):
        self.leaf_nodes.extend(nodes)

    def filter_nodes(self, max_nodes):
        # Keep only the top nodes based on their scores
        self.leaf_nodes = sorted(
            self.leaf_nodes, key=lambda node: node['score'], reverse=True
        )[:max_nodes]

    def get_nodes_to_expand(self, batch_size):
        # Get nodes that are not completed
        nodes_to_expand = [node for node in self.leaf_nodes if not node.get('is_completed', False)]
        # Get the top nodes to expand
        nodes_to_expand = sorted(nodes_to_expand, key=lambda node: node['score'], reverse=True)[:batch_size]
        return nodes_to_expand

    def remove_nodes(self, nodes_to_remove):
        # Remove nodes from leaf_nodes
        self.leaf_nodes = [node for node in self.leaf_nodes if node not in nodes_to_remove]

    def find_best_node(self):
        if not self.leaf_nodes:
            return None
        # Separate completed and incomplete nodes
        completed_nodes = [node for node in self.leaf_nodes if node.get('is_completed', False)]
        if completed_nodes:
            # Find the completed node with the highest score
            best_node = max(completed_nodes, key=lambda node: node['score'])
        else:
            # If no completed nodes, find the best incomplete node
            best_node = max(self.leaf_nodes, key=lambda node: node['score'])
        return best_node['tokens']
