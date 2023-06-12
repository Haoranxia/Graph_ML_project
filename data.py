def extract_info_pyg(graph_pyg):

    x_cat = graph_pyg.category
    y_geom = graph_pyg.geometry
    edge_index = graph_pyg.edge_index

    return x_cat, y_geom, edge_index


# dataset and dataloader
class PolyGraphDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train'):
        self.graph_path = os.path.join(path, 'graphs')
        self.correct_ids = torch.load(os.path.join(path, f'correct_ids.pickle'))[mode]

        # TODO: include graph transformations if necessary
        # self.graph_transform = graph_transform

    def __getitem__(self, index):

        # get floor plan identity
        floor_id = self.correct_ids[index]

        # get graph
        graph_pyg = torch.load(os.path.join(self.graph_path, f'{floor_id}.pickle'))

        return graph_pyg

    def __len__(self):
        return len(self.correct_ids)