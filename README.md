# GNN_Spatial
The repository is the PyTorch implamentation of Graph Convolutional Networks and Graph Attention Networks for spatial information extraction from time-series flow data (PeMS04, METR_LA).

## Run the demo
Jamp to `/config/` and edit the corresponding `.conf` as needed.
- `nums_of_nodes`: number of the nodes on the road network
- `len`: length of time windows for prediction
- `time_interval`: time period of each time window
- `batch_size`: number of time slots per batch
- `learning_rate` 
- `epochs`
- `hidden_dim`: the hidden dimensions of the middle layers
- `training_days`: length of training data(days)
- `validation_days`: length of validation data(days)
- `test_days`: length of test data(days)

And then to run the demo

`cd module
python trainer.py`

## Models
You can choose between the following models from `baseline.py`:

- `GCN3`: Three-layered GCN
- `GCN2`: Two-layered GCN
- `bGCN`: Graph convolutional network (Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016)
- `ChebConv`: One-layer Chebyshev polynomial version of graph convolutional network as described in (Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, NIPS 2016)
- `ChebNet`: Two-layered Chebyshev polynomial version of graph convolutional network
- `GraphAttentionLayer`: One-layer graph attention layer with self-attention mechanism
- `GATSubNet`: One-layer graph attention layer with multi-attention mechanism (the time-series input is splated into flat one-dimensional array as [B, N, T * C])
- `GATNet2`: One-layer graph attention layer with multi-attention mechanism (apply attention mechanism on T*[B, N, C] tensor, the input should be [B, N, C])

