import pickle
import torch as th 
import os 
import torch_geometric as pyg 
from torch_geometric.data import Data, Batch
import numpy as np

SUBTYPES_TO_REMOVE = [
 'WINTERGARTEN',
 'OUTDOOR_VOID',
 'MEETING_ROOM',
 'SALESROOM',
 'OPEN_PLAN_OFFICE',
 'PRAM',
 'ARCHIVE',
 'WAITING_ROOM',
 'OIL_TANK',
 'TRANSPORT_SHAFT',
 'AIR',
 'RECEPTION_ROOM',
 'FACTORY_ROOM',
 'WATER_SUPPLY',
 'COUNTER_ROOM',
 'TEACHING_ROOM',
 'BREAK_ROOM',
 'LOGISTICS',
 'RADATION_THERAPY',
 'WORKSHOP',
 'DEDICATED_MEDICAL_ROOM',
 'GAS',
 'PHYSIO_AND_REHABILITATION',
 'ARCADE']


SUBTYPE_MAPPING = {
    'ROOM': 'Bedroom',
    'BEDROOM': 'Bedroom',
    'KITCHEN': 'Kitchen-Dining',
    'DINING': 'Kitchen-Dining',
    'KITCHEN_DINING': 'Kitchen-Dining',
    'LIVING_ROOM': 'Living-Room',
    'LIVING_DINING': 'Living-Room',
    'RECEPTION_ROOM': 'Living-Room',
    'BATHROOM': 'Bathroom',
    'TOILET': 'Bathroom',
    'SHOWER': 'Bathroom',
    'BATHTUB': 'Bathroom',
    'CORRIDOR': 'Corridor',
    'CORRIDORS_AND_HALLS': 'Corridor',
    'LOBBY': 'Corridor',
    'OFFICE': 'Office',
    'OFFICE_SPACE': 'Office',
    'OPEN_PLAN_OFFICE': 'Office',
    'STAIRS': 'Stairs-Ramp',
    'STAIRCASE': 'Stairs-Ramp',
    'RAMP': 'Stairs-Ramp',
    'BASEMENT': 'Basement',
    'BASEMENT_COMPARTMENT': 'Basement',
    'COLD_STORAGE': 'Basement',
    'GARAGE': 'Garage',
    'BIKE_STORAGE': 'Garage',
    'PRAM_AND_BIKE_STORAGE_ROOM': 'Garage',
    'CARPARK': 'Garage',
    'WORKSHOP': 'Workshop',
    'FACTORY_ROOM': 'Workshop',
    'BALCONY': 'Outdoor-Area',
    'GARDEN': 'Outdoor-Area',
    'TERRACE': 'Outdoor-Area',
    'PATIO': 'Outdoor-Area',
    'OUTDOOR_VOID': 'Outdoor-Area',
    'WAREHOUSE': 'Warehouse-Logistics',
    'LOGISTICS': 'Warehouse-Logistics',
    'ARCHIVE': 'Archive-Records',
    'RECORDS': 'Archive-Records',
    'MEETING_ROOM': 'Meeting-Salesroom',
    'SALESROOM': 'Meeting-Salesroom',
    'SHOWROOM': 'Meeting-Salesroom'
}


""" 
Categories as a dictionary with their indices for onehot encoding
"""
CATEGORY_DICT = {
 'Remaining': 0,
 'Bathroom': 1,
 'Kitchen-Dining': 2,
 'Bedroom': 3,
 'Corridor': 4,
 'Stairs-Ramp': 5,
 'Outdoor-Area': 6,
 'Living-Room': 7,
 'Basement': 8,
 'Office': 9,
 'Garage': 10,
 'Warehouse-Logistics': 11,
 'Meeting-Salesroom': 12
}


def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


# helper function(s)
def extract_info_pyg(graph_pyg):

    x_cat = graph_pyg.category
    y_geom = graph_pyg.geometry
    edge_index = graph_pyg.edge_index

    return x_cat, y_geom, edge_index


# dataset and dataloader
class PolyGraphDataset(th.utils.data.Dataset):
    def __init__(self, path, mode='train'):
        self.graph_path = os.path.join(path, 'graphs')
        self.correct_ids = th.load(os.path.join(path, f'correct_ids.pickle'))[mode]

        # TODO: include graph transformations if necessary
        # self.graph_transform = graph_transform

    def __getitem__(self, index):

        # get floor plan identity
        floor_id = self.correct_ids[index]

        # get graph
        graph_pyg = th.load(os.path.join(self.graph_path, f'{floor_id}.pickle'))
        print(graph_pyg)
        del graph_pyg.walls
        del graph_pyg.door_geometry

        return graph_pyg

    def __len__(self):
        return len(self.correct_ids)
    

class Transformed_PolyGraphDataset(th.utils.data.Dataset):
    def __init__(self, path, mode='train'):
        self.graph_path = os.path.join(path, 'graphs')
        self.correct_ids = th.load(os.path.join(path, f'correct_ids.pickle'))[mode]

        # TODO: include graph transformations if necessary
        # self.graph_transform = graph_transform

    def __getitem__(self, index):
        # get floor plan identity
        floor_id = self.correct_ids[index]

        # get graph
        graph_pyg = th.load(os.path.join(self.graph_path, f'{floor_id}.pickle'))
        del graph_pyg.walls
        del graph_pyg.centroid 
        del graph_pyg.connectivity 

        # door = getattr(graph_pyg, "door-geometry")  
        # del door

        # convert edge_idx, geometry, categories to Tensors
        graph_pyg.category = onehot(graph_pyg.category)
        # print(graph_pyg.geometry)
        graph_pyg.geometry = pad_geometry(graph_pyg.geometry, MAX_POLYGONS=30)      # TODO: Read this from file?

        # print("dataloader gotten type:", type(graph_pyg))
        return graph_pyg

    def __len__(self):
        return len(self.correct_ids)
    

def onehot(categories):
    """ 
    category: must match CATEGORIES as string
    """
    onehot_encoding = th.zeros((len(categories), len(CATEGORY_DICT)))
    for i, cat in enumerate(categories):
        onehot_encoding[i][CATEGORY_DICT[cat]] = 1
    return onehot_encoding


def pad_geometry(geometry, MAX_POLYGONS):
    """ 
    geometry:       list of np.ndarray representing geometries
    MAX_POLYGONS:   int
    """
    padded_geometry = th.zeros((len(geometry), MAX_POLYGONS * 2), dtype=th.float)

    # geometry contains all the polygons for each room
    for i, polygon in enumerate(geometry):
        # polygon could be a list of tuples [(x1, y1), (x2, y2), ...]
        if isinstance(polygon, list):
            xs = [x for x, _ in polygon]
            ys = [y for _, y in polygon]    
            polygon = th.FloatTensor([xs, ys]).squeeze()

        # polygon is of shape (nr of polygons, 2) (hopefully)
        max = polygon.shape[0]

        # Check if we don't have too many polygons 
        if max > MAX_POLYGONS:
            # take polygons evenly spaced in the polygon
            # TODO: Improve this?
            incr = int(max / MAX_POLYGONS)
            indices = [i * incr for i in range(MAX_POLYGONS)]
            new_poly = polygon[indices, :]
            polygon = th.FloatTensor(new_poly)

            max = MAX_POLYGONS
        else:
            polygon = th.FloatTensor(polygon)

        # Fill tensor
        padded_geometry[i, :max] = polygon[:, 0]
        padded_geometry[i, MAX_POLYGONS:MAX_POLYGONS + max] = polygon[:, 1]
    
    return padded_geometry