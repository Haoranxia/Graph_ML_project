# Graph_ML_project

# Current idea
GAN 

Generator
- Input:
    1. Adjacency Matrix
    2. Node features: room coordinates, room type (one hot?), centroid
    3. Bounding constraint (optional)

Model architecture:


- Output:
    1. Node features: room coordinates, room type (one hot?)

Loss:

   

Discriminator
- Input:
    1. Either from Data or Generator: Node features (room coordinates, room type)
    2. Adjacency Matrix 

Model architecture:

- Output: 
    1. Fake or Real

Loss: 
    WGAN-GP Loss (house-gan)


# Questions
1. How to encode polygon sequence
2. How to properly encode the categories room types (embedding, hashing, ...)
3. Loss function in general
