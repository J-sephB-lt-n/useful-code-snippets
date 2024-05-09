"""
TAGS: connect|connection|connections|graph|path|paths|rustworkx|simple path|simple paths|traverse|undirected|undirected graph
DESCRIPTION: Enumerate all paths between nodes in an undirected graph
"""

# pip install matplotlib
# pip install rustworkx

"""
a1---a2---a4---a5
|         |
a3--------/

b1----b2----b3

      c1
      |
c2---c3----c4
      |    
      c5
"""

import matplotlib.pyplot as plt
import rustworkx as rx
import rustworkx.visualization

gph = rx.PyGraph()

node_name_to_idx: dict[str, int] = {}
node_idx_to_name: dict[int, str] = {}
for node_name in (
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "b1",
    "b2",
    "b3",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
):
    assigned_idx = gph.add_node(node_name)
    node_name_to_idx[node_name] = assigned_idx
    node_idx_to_name[assigned_idx] = node_name

edge_name_to_idx: dict[str, int] = {}
edge_name_to_idx["a1-a3"] = gph.add_edge(
    node_name_to_idx["a1"], node_name_to_idx["a3"], None
)
edge_name_to_idx["a1-a2"] = gph.add_edge(
    node_name_to_idx["a1"], node_name_to_idx["a2"], None
)
edge_name_to_idx["a2-a4"] = gph.add_edge(
    node_name_to_idx["a2"], node_name_to_idx["a4"], None
)
edge_name_to_idx["a3-a4"] = gph.add_edge(
    node_name_to_idx["a3"], node_name_to_idx["a4"], None
)
edge_name_to_idx["a4-a5"] = gph.add_edge(
    node_name_to_idx["a4"], node_name_to_idx["a5"], None
)

edge_name_to_idx["b1-b2"] = gph.add_edge(
    node_name_to_idx["b1"], node_name_to_idx["b2"], None
)
edge_name_to_idx["b2-b3"] = gph.add_edge(
    node_name_to_idx["b2"], node_name_to_idx["b3"], None
)

edge_name_to_idx["c1-c3"] = gph.add_edge(
    node_name_to_idx["c1"], node_name_to_idx["c3"], None
)
edge_name_to_idx["c2-c3"] = gph.add_edge(
    node_name_to_idx["c2"], node_name_to_idx["c3"], None
)
edge_name_to_idx["c3-c4"] = gph.add_edge(
    node_name_to_idx["c3"], node_name_to_idx["c4"], None
)
edge_name_to_idx["c3-c5"] = gph.add_edge(
    node_name_to_idx["c3"], node_name_to_idx["c5"], None
)

rustworkx.visualization.mpl_draw(gph, with_labels=True)
plt.show()

all_pairs_all_simple_paths = rx.all_pairs_all_simple_paths(
    gph,
    cutoff=4,  # max number of steps to allow
)
for src_node_idx, many_paths in all_pairs_all_simple_paths.items():
    print(node_idx_to_name[src_node_idx])
    for dest_node_idx, paths in many_paths.items():
        print("\t", node_idx_to_name[dest_node_idx])
        for path in paths:
            print("\t\t", "->".join([node_idx_to_name[idx] for idx in path]))
