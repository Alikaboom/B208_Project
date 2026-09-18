import osmnx as ox
import requests
import heapq
from collections import deque
import random
import time
import tracemalloc

ox.settings.log_console = False
ox.settings.use_cache = True

print("downloading nuremberg street graph (this might take a minute)...")
place = "Nuremberg, Germany"
G = ox.graph_from_place(place, network_type='drive')

print(f"graph loaded! nodes: {len(G.nodes)}, edges: {len(G.edges)}")

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


def bfs_shortest_path(graph, source, target):
    visited = set([source])
    previous = {node: None for node in graph.nodes}
    queue = deque([source])
    
    while queue:
        current = queue.popleft()
        
        if current == target:
            break
            
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                previous[neighbor] = current
                queue.append(neighbor)
                
    path = []
    curr = target
    tot_dist = 0.0
    
    while curr is not None:
        path.insert(0, curr)
        prev_node = previous[curr]
        if prev_node is not None:
            # sum physical distance
            edge_data = graph.get_edge_data(prev_node, curr)
            weight = min([d['length'] for d in edge_data.values()])
            tot_dist += weight
        curr = prev_node
        
    return path, tot_dist


def dijkstra_heap(graph, source, target):
    nodes = list(graph.nodes)
    distances = {node: float('inf') for node in nodes}
    previous = {node: None for node in nodes}
    distances[source] = 0
    pq = [(0, source)]
    
    while pq:
        dist, current = heapq.heappop(pq)
        
        if dist > distances[current]:
            continue
            
        if current == target:
            break
            
        for neighbor in graph.neighbors(current):
            # osmnx multidigraph edges
            edge_data = graph.get_edge_data(current, neighbor)
            weight = min([d['length'] for d in edge_data.values()])
            
            alt = dist + weight
            if alt < distances[neighbor]:
                distances[neighbor] = alt
                previous[neighbor] = current
                heapq.heappush(pq, (alt, neighbor))
                
    path = []
    curr = target
    while curr is not None:
        path.insert(0, curr)
        curr = previous[curr]
        
    return path, distances[target]


if __name__ == "__main__":
    print("generating 5 emergency scenarios...")
    scenarios = random.sample(list(G.nodes), 5)
    
    for i, start_node in enumerate(scenarios, 1):
        print(f"\n--- scenario {i}: routing from node {start_node} to hospital {target_node} ---")
        
        tracemalloc.start()
        start_time = time.perf_counter()
        bfs_path, bfs_dist = bfs_shortest_path(G, start_node, target_node)
        bfs_time = time.perf_counter() - start_time
        bfs_mem = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
        
        tracemalloc.start()
        start_time = time.perf_counter()
        heap_path, heap_dist = dijkstra_heap(G, start_node, target_node)
        heap_time = time.perf_counter() - start_time
        heap_mem = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
        
        print(f"bfs        -> dist: {bfs_dist:.2f}m, time: {bfs_time:.4f}s, mem: {bfs_mem:.1f}KB, nodes: {len(bfs_path)}")
        print(f"dijkstra   -> dist: {heap_dist:.2f}m, time: {heap_time:.4f}s, mem: {heap_mem:.1f}KB, nodes: {len(heap_path)}")
        
        ox.plot_graph_route(G, heap_path, route_color='r', route_linewidth=4, node_size=0, show=False, save=True, filepath=f"scenario_{i}_route.png")
