
import math, random, time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


#Parameters used

file_path = "C:/Users/chitr/Desktop/IM-C/Influence-Maximization/tagBased/Crime_Data_from_2020_to_Present.csv"

k = 5                    # number of seeds to select
R = 400                  # Monte Carlo runs -> Find expected spread if the node is choosen->runs R times
lambda_param = 0.7       # balance between marginal gain and diversity
candidate_top_n = None   # set to integer e.g. 80 to prune candidates. None -> use all nodes
random_seed = 42

random.seed(random_seed)


df = pd.read_csv(file_path)

# Column names used in dataset 
crime_code_cols = ["Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4"]
desc_col = "Crm Cd Desc" #Primary Edge
area_col = "AREA NAME" #Secondary Edge
date_col = "DATE OCC" #Temporal Edge

# If defined column not in dataset
for c in crime_code_cols + [desc_col, area_col, date_col, "Crm Cd"]:
    if c not in df.columns:
        raise RuntimeError(f"Expected column '{c}' not found in CSV. Found columns: {df.columns.tolist()}")

# Map Crm Cd numeric code -> description
code_to_desc = df[["Crm Cd", desc_col]].dropna().drop_duplicates().set_index("Crm Cd")[desc_col].to_dict()

# Work on copy
tag_df = df[crime_code_cols + [desc_col, area_col, date_col]].copy()

# Map Crm Cd 1..4 to descriptions (if present)
for col in crime_code_cols:
    tag_df[col] = tag_df[col].map(code_to_desc)

tag_df[desc_col] = tag_df[desc_col].astype(str).str.strip()

# Build undirected weighted graph
G = nx.Graph()
def add_tag_edge(u, v, w=1):
    if u is None or v is None: return
    if pd.isna(u) or pd.isna(v): return
    u = str(u).strip(); v = str(v).strip()
    if not u or not v or u == v: return
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
    else:
        G.add_edge(u, v, weight=w)

# Primary Edge -> Defined directly in dataset
for idx, row in tag_df.iterrows():
    tags = []
    for c in crime_code_cols:
        val = row.get(c)
        if pd.notna(val):
            sval = str(val).strip()
            if sval and sval.lower() != 'nan':
                tags.append(sval)
    tags = list(dict.fromkeys(tags))
    if len(tags) > 1:
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                add_tag_edge(tags[i], tags[j], w=1)



# Secondary Edge

# Two crimes co-occur in the same AREA NAME on the same DATE OCC
tag_df['DateOnly'] = pd.to_datetime(tag_df[date_col], errors='coerce').dt.date
for (area, date), group in tag_df.groupby([area_col, 'DateOnly']):
    tags = group[desc_col].dropna().astype(str).str.strip().unique()
    tags = [t for t in tags if t and t.lower() != 'nan']
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            add_tag_edge(tags[i], tags[j], w=1)


# Temporal Edge

# Combine DATE OCC + TIME OCC into a single datetime column
if "TIME OCC" in df.columns:
    # Convert TIME OCC (HHMM) into hours + minutes
    tag_df['TimeOcc'] = pd.to_numeric(df["TIME OCC"], errors='coerce')
    tag_df['HourOcc'] = tag_df['TimeOcc'].fillna(0).astype(int).astype(str).str.zfill(4).str[:2].astype(int)
    tag_df['MinuteOcc'] = tag_df['TimeOcc'].fillna(0).astype(int).astype(str).str.zfill(4).str[2:].astype(int)

    tag_df['DateTimeOcc'] = pd.to_datetime(tag_df[date_col], errors='coerce')
    tag_df['DateTimeOcc'] = tag_df['DateTimeOcc'] + \
                            pd.to_timedelta(tag_df['HourOcc'], unit='h') + \
                            pd.to_timedelta(tag_df['MinuteOcc'], unit='m')

    time_window_hours = 1  # configurable

    for area, group in tag_df.groupby(area_col):
        group = group.dropna(subset=['DateTimeOcc'])
        group = group[['DateTimeOcc', desc_col]].dropna().sort_values('DateTimeOcc')

        times = group['DateTimeOcc'].tolist()
        tags = group[desc_col].astype(str).str.strip().tolist()

        # sliding window to avoid O(n^2)
        j = 0
        for i in range(len(times)):
            while j < len(times) and (times[j] - times[i]).total_seconds() <= time_window_hours * 3600:
                if i != j:
                    add_tag_edge(tags[i], tags[j], w=1)
                j += 1
else:
    print("TIME OCC column not found — skipping temporal edges")



# Make sure isolated tags (if any) are included as nodes
for t in tag_df[desc_col].dropna().astype(str).str.strip().unique():
    if t and t.lower() != 'nan' and t not in G:
        G.add_node(t)

# Assign influence probabilities on edges (undirected -> treat as both directions)
max_w = max((d['weight'] for _,_,d in G.edges(data=True)), default=1)
for u,v,d in G.edges(data=True):
    w = d.get('weight', 1)
    # tuned mapping: base 0.05 up to ~0.3
    p = 0.05 + 0.25 * (w / max_w)
    G[u][v]['p'] = p

# Build adjacency-prob mapping for IC simulation -> For influence calculation
adj_prob = defaultdict(list)
for u,v,d in G.edges(data=True):
    p = d.get('p', 0.05)
    adj_prob[u].append((v,p))
    adj_prob[v].append((u,p))

# IC simulation (single run)
def run_ic(seed_set, adj_local):
    activated = set(seed_set)
    newly_active = set(seed_set)
    while newly_active:
        next_active = set()
        for u in newly_active:
            for v,p in adj_local.get(u, []):
                if v not in activated and random.random() <= p:
                    next_active.add(v)
        newly_active = next_active - activated
        activated |= newly_active
    return activated

# Monte-Carlo expected spread (with caching)
sigma_cache = {}
def estimate_spread(seed_set, R_sim=200):
    key = frozenset(seed_set)
    if key in sigma_cache:
        return sigma_cache[key]
    if not seed_set:
        sigma_cache[key] = 0.0
        return 0.0
    tot = 0
    for _ in range(R_sim):
        tot += len(run_ic(seed_set, adj_prob))
    val = tot / R_sim
    sigma_cache[key] = val
    return val

# Precompute all-pairs shortest path lengths for diversity (unweighted)
n_nodes = G.number_of_nodes()
all_shortest = dict(nx.all_pairs_shortest_path_length(G))
def dist(u,v):
    if u == v: return 0
    d = all_shortest.get(u, {}).get(v, None)
    return d if d is not None else n_nodes

# Candidate pruning: optionally consider only top-N nodes by degree
if candidate_top_n is not None:
    deg_sorted = sorted(G.degree(), key=lambda x:x[1], reverse=True)
    candidate_nodes = [n for n,_ in deg_sorted[:candidate_top_n]]
else:
    candidate_nodes = list(G.nodes())

# Greedy IM-C (candidate-pruned)
def im_c_greedy(k, lambda_param=0.7, R_sim=200, candidates=None):
    S = set()
    S_info = []
    for i in range(k):
        sigma_S = estimate_spread(S, R_sim=R_sim) if S else 0.0
        best = (None, -1e9, 0.0, 0.0, 0.0)  # (node, score, delta, D, sigma_Sv)
        iter_nodes = candidates if candidates is not None else G.nodes()
        for v in iter_nodes:
            if v in S: continue
            sigma_Sv = estimate_spread(S.union({v}), R_sim=R_sim)
            delta = (sigma_Sv - sigma_S) / sigma_Sv if sigma_Sv > 0 else 0.0
            if not S:
                D_vs = 1.0
            else:
                avg_d = sum(dist(v, s) for s in S) / len(S)
                D_vs = 1.0 / (1.0 + avg_d)
            score = lambda_param * delta + (1 - lambda_param) * D_vs
            if score > best[1]:
                best = (v, score, delta, D_vs, sigma_Sv)
        if best[0] is None:
            break
        node, score, delta, D_vs, sigma_Sv = best
        S.add(node)
        S_info.append({'node':node, 'score':score, 'delta':delta, 'D':D_vs, 'sigma_Sv':sigma_Sv})
        print(f"Selected {len(S)}/{k}: {node}  score={score:.4f}  Δ={delta:.4f}  D={D_vs:.4f}  est_spread={sigma_Sv:.2f}")
    return S_info


# Run IM-C 

start = time.time()
print("Graph built: nodes=%d edges=%d" % (G.number_of_nodes(), G.number_of_edges()))
print("Running IM-C (this can be slow; demo R=%d). Increase R for better accuracy." % R)
seeds = im_c_greedy(k=k, lambda_param=lambda_param, R_sim=R, candidates=candidate_nodes)
end = time.time()
print("Done in %.1fs" % (end - start))

final_seed_list = [s['node'] for s in seeds]
final_spread = estimate_spread(set(final_seed_list), R_sim=R)
print("\nFinal seeds (k=%d):" % k)
for idx, s in enumerate(seeds, 1):
    print(f"{idx}. {s['node']} score={s['score']:.4f} Δ={s['delta']:.4f} D={s['D']:.4f} est_spread_if_added={s['sigma_Sv']:.2f}")
print(f"\nEstimated spread of final seeds: {final_spread:.2f} / {n_nodes} nodes")

# Visualization: draw full graph and highlight seeds
plt.figure(figsize=(18,14))
pos = nx.spring_layout(G, k=0.25, iterations=200, seed=random_seed)
deg = dict(G.degree()); max_deg = max(deg.values()) if deg else 1
node_sizes = [60 + 400*(deg[n]/max_deg) for n in G.nodes()]
node_colors = ['red' if n in final_seed_list else 'skyblue' for n in G.nodes()]
edge_widths = [max(0.3, 2.0* (G[u][v]['weight'] / max_w)) for u,v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=7)
plt.title("Crime Tag Co-occurrence Network — IM-C seeds in RED", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
