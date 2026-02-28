import networkx as nx
import numpy as np
from numba import njit

# ==========================================================
# NUMBA JIT COMPILATION
# Converts this math into bare-metal C code for maximum speed.
# fastmath=True allows the CPU to drop strict IEEE compliance for faster floating-point math.
# ==========================================================
@njit(fastmath=True)
def fast_haversine(lat1, lon1, lat2, lon2):
    """C-level Haversine distance calculation."""
    R = 6371000.0  # Earth radius in meters
    
    # Convert degrees to radians
    p1 = lat1 * np.pi / 180.0
    p2 = lat2 * np.pi / 180.0
    dp = (lat2 - lat1) * np.pi / 180.0
    dl = (lon2 - lon1) * np.pi / 180.0
    
    a = np.sin(dp / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    return R * c

def find_energy_route_astar(graph, orig_node, dest_node):
    """Finds the most energy-efficient route using pre-calculated ML edge weights and Numba."""
    
    # 1. MEMORY FLATTENING: Pre-extract all coordinates.
    # This completely eliminates slow graph.nodes[u] dictionary lookups during the A* loop.
    coords = {n: (data['y'], data['x']) for n, data in graph.nodes(data=True)}
    target_lat, target_lon = coords[dest_node]

    # 2. THE OPTIMIZED HEURISTIC
    def heuristic_energy(u, v):
        # Target node 'v' is ignored here because we always compare 'u' to the final destination
        u_lat, u_lon = coords[u]
        
        # Fire the Numba machine-code function
        dist_m = fast_haversine(u_lat, u_lon, target_lat, target_lon)
        
        # Assume a highly efficient 0.1 Wh per meter as the optimistic heuristic cost
        return dist_m * 0.1

    # 3. EXECUTE SEARCH
    return nx.astar_path(
        graph, 
        orig_node, 
        dest_node, 
        heuristic=heuristic_energy,
        weight='ml_energy_cost'
    )
    
def find_shortest_route(graph, orig_node, dest_node):
    """Finds the shortest physical route using standard Dijkstra."""
    return nx.shortest_path(graph, orig_node, dest_node, weight='length')