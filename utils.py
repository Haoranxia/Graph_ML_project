import pickle
import torch as th 
import os 
import torch_geometric as pyg 
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm.auto import tqdm


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

ZONING_MAPPING = {
    'ROOM': 'Zone1',
    'BEDROOM': 'Zone1',
    'KITCHEN': 'Zone2',
    'DINING': 'Zone2',
    'KITCHEN_DINING': 'Zone2',
    'LIVING_ROOM': 'Zone2',
    'LIVING_DINING': 'Zone2',
    'RECEPTION_ROOM': 'Remaining',
    'BATHROOM': 'Zone3',
    'TOILET': 'Zone3',
    'SHOWER': 'Zone3',
    'BATHTUB': 'Zone3',
    'CORRIDOR': 'Zone2',
    'CORRIDORS_AND_HALLS': 'Zone2',
    'LOBBY': 'Remaining',
    'OFFICE': 'Remaining',
    'OFFICE_SPACE': 'Remaining',
    'OPEN_PLAN_OFFICE': 'Remaining',
    'STAIRS': 'Zone3',
    'STAIRCASE': 'Zone3',
    'RAMP': 'Remaining',
    'BASEMENT': 'Remaining',
    'BASEMENT_COMPARTMENT': 'Remaining',
    'COLD_STORAGE': 'Remaining',
    'GARAGE': 'Remaining',
    'BIKE_STORAGE': 'Remaining',
    'PRAM_AND_BIKE_STORAGE_ROOM': 'Remaining',
    'CARPARK': 'Remaining',
    'WORKSHOP': 'Remaining',
    'FACTORY_ROOM': 'Remaining',
    'BALCONY': 'Balcony',
    'GARDEN': 'Remaining',
    'TERRACE': 'Remaining',
    'PATIO': 'Remaining',
    'OUTDOOR_VOID': 'Remaining',
    'WAREHOUSE': 'Remaining',
    'LOGISTICS': 'Remaining',
    'ARCHIVE': 'Remaining',
    'RECORDS': 'Remaining',
    'MEETING_ROOM': 'Remaining',
    'SALESROOM': 'Remaining',
    'SHOWROOM': 'Remaining'
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

"""
Slightly modified dataset class from original
"""
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
        del graph_pyg.door_geometry

        # convert edge_idx, geometry, categories to Tensors
        graph_pyg.category = onehot(graph_pyg.category)
        graph_pyg.geometry = pad_geometry(graph_pyg.geometry, MAX_POLYGONS=30)      # TODO: Read this from file?

        return graph_pyg

    def __len__(self):
        return len(self.correct_ids)
    

def onehot(categories):
    """ 
    category: must match CATEGORIES as string
    """
    onehot_encoding = th.zeros((len(categories), len(CATEGORY_DICT)))
    for i, cat in enumerate(categories):
        onehot_encoding[i, CATEGORY_DICT[cat]] = 1
    return onehot_encoding


def pad_geometry(geometry, MAX_POLYGONS):
    """ 
    geometry:       list of np.ndarray representing geometries
    MAX_POLYGONS:   int

    return:         tensor of size (nr. geometries, 60)
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
            # TODO: Improve this, how to select 'correct' points?
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


def load_pickle(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()
