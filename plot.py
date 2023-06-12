import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import networkx as nx

FS = 10
SUBTYPES_NAMES = [
     'Remaining',
     'Bathroom',
     'Kitchen-Dining',
     'Bedroom',
     'Corridor',
     'Stairs-Ramp',
     'Outdoor-Area',
     'Living-Room',
     'Basement',
     'Office',
     'Garage',
     'Warehouse-Logistics',
     'Meeting-Salesroom'
]


def plot_polygon(ax, poly, **kwargs):
    x, y = poly.exterior.xy
    ax.fill(x, y, **kwargs)
    return


def plot_floorplan(ax, areas, area_types, doors=False, walls=False, classes=SUBTYPES_NAMES, colorset='tab20', **kwargs):

    cmap = get_cmap(colorset)

    for area, area_type in zip(areas, area_types):
        try:
            np.isnan(area_type)
            color_index = len(classes) + 1
        except:
            color_index = classes.index(area_type)
        c=np.array(cmap(color_index)).reshape(1,4)
        plot_polygon(ax, area, fc=c, ec=c, **kwargs)

    if walls:
        for wall in walls:
            plot_polygon(ax, wall, fc='#72246c', ec='#72246c', **kwargs)
    if doors:
        for door in doors[0]:
            plot_polygon(ax, door, fc='red', ec='red', **kwargs)

        for door in doors[1]:
            plot_polygon(ax, door, fc='orange', ec='orange', **kwargs)


def plot_graph(G, ax, c_node='red', c_edge=['white', 'red', 'red', 'yellow'], dw_edge=False, pos=None, node_size=10,
               edge_size=10):

    """
    Plots the adjacency or access graph of a floor plan's corresponding graph structure.
    """

    # position
    if pos is None:
        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c_node, ax=ax)

    # edges
    if dw_edge:
        epass = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'passage']
        edoor = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'door']
        efront = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'entrance']
        # red full for passage, red dashed for door, yellow dashed for front
        nx.draw_networkx_edges(G, pos, edgelist=epass, edge_color=c_edge[1],
                               width=edge_size, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edoor, edge_color=c_edge[2],
                               width=edge_size, style="dashed", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=efront, edge_color=c_edge[3],
                               width=edge_size, style="-.", ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, edge_color=c_edge[0],
                               width=edge_size, ax=ax)

    ax.axis('off')