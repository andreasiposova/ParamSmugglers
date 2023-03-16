import networkx as nx


def visualize_graph(net):
    # create a directed graph
    G = nx.DiGraph()

    # add nodes for the biases
    for name, param in net.named_parameters():
        if 'bias' in name:
            G.add_node(name)

    # add edges for the weights
    for name, param in net.named_parameters():
        if 'weight' in name:
            src, dst = name.split('.')
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    weight = param[i, j].item()
                    G.add_edge(f'{src}.bias', f'{dst}.bias', weight=weight)

    # plot the graph using Matplotlib
    pos = nx.spring_layout(G)
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=500, node_color='r')
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color='b')
    nx.draw_networkx_labels(G, pos, labels={node: node.split('.')[1] for node in G.nodes()})

