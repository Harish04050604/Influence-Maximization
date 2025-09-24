import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load dataset
# -------------------------------
file_path = r"C:\Pranav_Storage\Documents\College_Coding\final_yr_project_1\Influence-Maximization\contextBased\archive (7)\crime_dataset_india.csv"
df = pd.read_csv(file_path)

required_cols = ['City', 'Crime Description', 'Report Number']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise RuntimeError(f"Missing columns in CSV: {missing_cols}")

# -------------------------------
# Step 2: Build Graph
# -------------------------------
G = nx.Graph()
crime_grouped = df.groupby(['City', 'Crime Description']).size().reset_index(name='count')

for desc, group in crime_grouped.groupby('Crime Description'):
    cities = group['City'].tolist()
    counts = group['count'].tolist()
    for (city1, count1), (city2, count2) in combinations(zip(cities, counts), 2):
        weight = count1 + count2
        if G.has_edge(city1, city2):
            G[city1][city2]['weight'] += weight
            G[city1][city2]['descriptions'][desc] = count1 + count2
        else:
            G.add_edge(city1, city2, weight=weight, descriptions={desc: count1 + count2})

print(f"Graph built: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

# -------------------------------
# Step 3: Define metrics
# -------------------------------
def compute_influential_gain(node, selected_nodes, G):
    neighbors = set(G.neighbors(node)) - set(selected_nodes)
    return sum(G[node][nbr]['weight'] for nbr in neighbors)

def compute_diversity_score(node, selected_nodes, G):
    if not selected_nodes:
        return 1.0
    lengths = nx.single_source_shortest_path_length(G, node)
    distances = [lengths.get(s, len(G)) for s in selected_nodes]
    avg_distance = sum(distances) / len(distances)
    return 1 / (1 + avg_distance)

lambda_param = 0.7  # weight for influence vs diversity
k = 5

# -------------------------------
# Step 4: Select top-k seeds uniquely
# -------------------------------
seeds_info = []
used_nodes = set()

nodes_remaining = list(G.nodes())

for i in range(k):
    best_node = None
    best_score = -1
    best_strongest = None
    best_weight = 0
    best_crime = None
    best_inf_gain = 0
    best_div_score = 0
    
    for node in nodes_remaining:
        if node in used_nodes:
            continue
        
        inf_gain = compute_influential_gain(node, list(used_nodes), G)
        div_score = compute_diversity_score(node, list(used_nodes), G)
        overall_score = lambda_param * inf_gain + (1 - lambda_param) * div_score

        # Determine strongest connection
        neighbors = [nbr for nbr in G.neighbors(node) if nbr not in used_nodes]
        if neighbors:
            strongest = max(neighbors, key=lambda nbr: G[node][nbr]['weight'])
            weight = G[node][strongest]['weight']
            crime = max(G[node][strongest]['descriptions'].items(), key=lambda x: x[1])[0]
        else:
            strongest, weight, crime = None, 0, None

        if overall_score > best_score:
            best_score = overall_score
            best_node = node
            best_strongest = strongest
            best_weight = weight
            best_crime = crime
            best_inf_gain = inf_gain
            best_div_score = div_score

    if best_node is None:
        break
    
    # Add both the seed and its strongest connection to used_nodes to ensure uniqueness
    used_nodes.add(best_node)
    if best_strongest:
        used_nodes.add(best_strongest)
    
    seeds_info.append({
        'node': best_node,
        'strongest_connection': best_strongest,
        'edge_weight': best_weight,
        'crime': best_crime,
        'overall_score': best_score,
        'influential_gain': best_inf_gain,
        'diversity_score': best_div_score
    })
    
    nodes_remaining = [n for n in nodes_remaining if n not in used_nodes]

# -------------------------------
# Step 5: Display final seed set
# -------------------------------
print("\nFinal Seed Set (City, Strongest Connected City, Crime Description, Edge Weight, Scores):\n")
for idx, s in enumerate(seeds_info, 1):
    print(f"{idx}. City: {s['node']}")
    print(f"   - Strongest Connection: {s['strongest_connection']} | Crime: {s['crime']} | Edge Weight={s['edge_weight']}")
    print(f"   - Influential Gain: {s['influential_gain']:.2f}")
    print(f"   - Diversity Score: {s['diversity_score']:.2f}")
    print(f"   - Overall Score: {s['overall_score']:.2f}\n")

# -------------------------------
# Step 6: Graph Visualization
# -------------------------------
plt.figure(figsize=(14,10))
pos = nx.spring_layout(G, seed=42)
node_colors = ['red' if n in [s['node'] for s in seeds_info] else 'skyblue' for n in G.nodes()]

edge_colors = []
edge_widths = []
for u, v, data in G.edges(data=True):
    if any((u==s['node'] and v==s['strongest_connection']) or (v==s['node'] and u==s['strongest_connection']) for s in seeds_info):
        edge_colors.append('red')
        edge_widths.append(3)
    else:
        edge_colors.append('gray')
        edge_widths.append(1)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=9)
plt.title("Crime Influence Graph — Seed Cities in RED", fontsize=16)
plt.axis('off')
plt.show()