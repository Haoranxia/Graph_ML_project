from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors


COLORS_ROOMTYPE = ['#1f77b4',
                      '#e6550d',
                      '#fd8d3c',
                      '#fdae6b',
                      '#fdd0a2',
                      '#72246c',
                      '#5254a3',
                      '#6b6ecf',
                      '#2ca02c',
                      '#000000',
                      '#1f77b4',
                      '#98df8a',
                      '#d62728']


COLOR_MAP_ROOMTYPE = mcolors.ListedColormap(COLORS_ROOMTYPE)
CMAP_ROOMTYPE = get_cmap(COLOR_MAP_ROOMTYPE)