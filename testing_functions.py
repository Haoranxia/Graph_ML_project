from model import gradient_penalty
from utils import Transformed_PolyGraphDataset

import torch_geometric as pyg 
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm.auto import tqdm
import torch as th

""" 
This file contains some functions that I used to test the model and its related functions with
"""

##################
## TESTING CODE ##
##################

from model import gradient_penalty

def test_model(dataset, generator, discriminator, NOISE_SIZE):
    print(dataset[0])
    test_list = [dataset[0] for _ in range(32)]
    test_batch = Batch.from_data_list(test_list)
    print(test_batch)

    real = test_batch

    noise = th.randn((len(real.category), NOISE_SIZE))
    print("noise: ", noise.shape)
    fake = generator(real, noise)     

    # fake.shape = (batch_size * nodes, output_features = 60)
    # We must turn this into appropriate (batch) input for the discriminator
    fake = Batch(geometry=fake, edge_index=real.edge_index, batch=real.batch)

    print("fake: ", fake.geometry.shape)
    print("real: ", real.geometry.shape)

    discriminator_fake = discriminator(fake)    # discriminator scores for fakes
    discriminator_real = discriminator(real)    # discriminator scores for reals

    print("fake: ", fake)
    print("real: ", real)
    print("discriminator score fake: ", discriminator_fake.shape)
    print("discriminator score real: ", discriminator_real.shape)

    gp = gradient_penalty(discriminator, real, fake)
    print(gp)


def test_dataloader(path, BATCH_SIZE):
    dataset = Transformed_PolyGraphDataset(path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    n_batches = len(dataloader)
    dataiter = iter(dataloader)

    for _ in tqdm(range(n_batches)):
        batch = next(dataiter)
        print(batch)