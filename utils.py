import pickle

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


def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object