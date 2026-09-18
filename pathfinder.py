import osmnx as ox
import requests

# setup caching so we don't spam the osm servers
ox.settings.log_console = False
ox.settings.use_cache = True

print("downloading munich street graph (this might take a minute)...")

# load the drivable network for our target area
place = "Schwabing, Munich, Germany"
G = ox.graph_from_address(place, dist=500, network_type='drive')

print(f"graph loaded! nodes: {len(G.nodes)}, edges: {len(G.edges)}")

# load api key securely
api_key = None
try:
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('GOOGLE_MAPS_API_KEY='):
                api_key = line.strip().split('=', 1)[1]
                break
except FileNotFoundError:
    pass

if not api_key:
    print("error: GOOGLE_MAPS_API_KEY not found in .env")
    exit()

print("searching for nearby hospitals via google maps API...")
url = "https://places.googleapis.com/v1/places:searchText"
headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": api_key,
    "X-Goog-FieldMask": "places.displayName,places.location"
}
payload = {"textQuery": f"hospital in {place}", "maxResultCount": 1}

res = requests.post(url, headers=headers, json=payload).json()
if 'places' not in res or len(res['places']) == 0:
    print("no hospitals found.")
    exit()

hospital = res['places'][0]
h_name = hospital.get('displayName', {}).get('text', 'Hospital')
h_lat = hospital['location']['latitude']
h_lng = hospital['location']['longitude']
print(f"found target: {h_name} at ({h_lat}, {h_lng})")

print("mapping hospital to the nearest graph node...")
target_node = ox.distance.nearest_nodes(G, h_lng, h_lat)
print(f"target mapped to node ID: {target_node}")

# --- PATHFINDING ALGORITHMS ---

import heapq

def dijkstra_array(graph, source, target):
    # node-indexed array implementation for O(n^2) efficiency
    nodes = list(graph.nodes)
    distances = {node: float('inf') for node in nodes}
    previous = {node: None for node in nodes}
    distances[source] = 0
    
    unvisited = set(nodes)
    
    while unvisited:
        # find the unvisited node with the smallest distance (O(n) scan)
        current = None
        min_dist = float('inf')
        for node in unvisited:
            if distances[node] < min_dist:
                min_dist = distances[node]
                current = node
                
        # if the remaining nodes are unreachable or we reached the target, stop
        if current is None or current == target:
            break
            
        unvisited.remove(current)
        
        # update neighbors
        for neighbor in graph.neighbors(current):
            if neighbor in unvisited:
                # osmnx graphs can have multiple edges between nodes, pick the shortest
                edge_data = graph.get_edge_data(current, neighbor)
                weight = min([d['length'] for d in edge_data.values()])
                
                alt = distances[current] + weight
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = current
                    
    return distances, previous

def dijkstra_heap(graph, source, target):
    # min-heap priority queue implementation for O(m log n) efficiency
    pass
