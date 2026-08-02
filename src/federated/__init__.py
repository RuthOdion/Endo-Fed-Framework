"""Federated learning client, server, and aggregation modules."""

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer, create_federated_clients
from src.federated.aggregation import fedavg_aggregate, fedprox_aggregate
