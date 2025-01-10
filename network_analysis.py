#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import os
import copy
import json
from collections import defaultdict, Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd

# Louvain community detection library
import community as community_louvain

###############################################################################
#                           OUTPUT DIRECTORY SETUP                             #
###############################################################################
OUTPUT_DIR = "analysed_topic_data"

# Constants
CATEGORY_FOLDER = "articles_by_year"  # Folder containing yearly JSON files
OUTPUT_GEXF_FULL = "topic_topic_graph/collective_topic_network_full.gexf"  # Output GEXF file
OUTPUT_GEXF_POS = "topic_topic_graph/collective_topic_network_pos.gexf"
OUTPUT_GEXF_NEG = "topic_topic_graph/collective_topic_network_neg.gexf"

def ensure_output_dir(directory=OUTPUT_DIR):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

###############################################################################
#                           LOADING GRAPHS FROM CSV                           #
###############################################################################
def load_full_graph_from_csv(csv_path, directed=False):
    """
    Loads the full network from CSV with columns:
    Topic1,Topic2,Edge Weight,Positive Sentiment,Negative Sentiment
    Returns a NetworkX Graph or DiGraph.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t1 = row['Topic1']
            t2 = row['Topic2']
            w = float(row['Edge Weight']) if row['Edge Weight'] else 0.0
            pos_s = float(row['Positive Sentiment']) if row['Positive Sentiment'] else 0.0
            neg_s = float(row['Negative Sentiment']) if row['Negative Sentiment'] else 0.0

            G.add_edge(t1, t2, 
                       weight=w, 
                       pos_sentiment=pos_s, 
                       neg_sentiment=neg_s)
            
            # if G[t1][t2]['weight'] < 0:
            #     print("added a negative edge")
            
    return G

def load_signed_graph_from_csv(csv_path, directed=False):
    """
    Loads a positive-only or negative-only network from CSV with columns:
    Topic1,Topic2,Edge Weight
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t1 = row['Topic1']
            t2 = row['Topic2']
            w = float(row['Edge Weight']) if row['Edge Weight'] else 0.0
            G.add_edge(t1, t2, weight=w)
    return G

###############################################################################
#                           PLOTTING GRAPHS                                   #
###############################################################################

def redistribute_neutral(pos, neg, neu):
    """
    Redistribute the neutral sentiment proportionally to positive and negative.
    """
    if pos + neg == 0:
        return neu / 2, neu / 2
    total = pos + neg
    pos_redistributed = pos + neu * (pos / total)
    neg_redistributed = neg + neu * (neg / total)
    return pos_redistributed, neg_redistributed

def process_all_json_files(folder_path, positive_only=False, negative_only=False):
    """
    Process all JSON files in the folder to build a collective graph.
    """
    G = nx.Graph()
    edge_data = defaultdict(lambda: {"count": 0, "positive": 0, "negative": 0})

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            with open(file_path, "r") as f:
                data = json.load(f)

                for article in data:
                    topics = [topic["name"] for topic in article.get("topics", [])]
                    sentiment = article.get("sentiment", {})

                    # Redistribute neutral sentiment
                    positive, negative = redistribute_neutral(
                        sentiment.get("positive", 0),
                        sentiment.get("negative", 0),
                        sentiment.get("neutral", 0)
                    )

                    if positive_only and positive <= negative:
                        continue
                    if negative_only and negative <= positive:
                        continue

                    # Create edges between topics if they have overlapping mentions in the article
                    for i in range(len(topics)):
                        for j in range(i + 1, len(topics)):
                            topic1, topic2 = sorted([topics[i], topics[j]])
                            edge_data[(topic1, topic2)]["count"] += 1
                            edge_data[(topic1, topic2)]["positive"] += positive
                            edge_data[(topic1, topic2)]["negative"] += negative

    # Add nodes and edges to the graph
    unique_topics = set(topic for edge in edge_data for topic in edge)
    for topic in unique_topics:
        G.add_node(topic)

    for (topic1, topic2), data in edge_data.items():
        total_count = abs(data["count"])
        G.add_edge(topic1, topic2, weight=total_count)

    return G, edge_data

def save_graph_to_gexf(G, output_file):
    """
    Save the graph in GEXF format.
    """
    nx.write_gexf(G, output_file)
    print(f"Graph saved to {output_file}")

###############################################################################
#                           COUNTING UNIQUE TOPICS (FROM JSON)                #
###############################################################################
def count_unique_topics(folder_path):
    """
    Counts the number of unique topics in all JSON files within a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        int: Count of unique topics.
        list: Sorted list of unique topics.
    """
    unique_topics = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    for item in data:
                        if "topics" in item:
                            for topic in item["topics"]:
                                unique_topics.add(topic["name"])
                except json.JSONDecodeError:
                    pass
    return len(unique_topics), sorted(unique_topics)

###############################################################################
#                       SENTIMENT BALANCED TRIANGLES                          #
###############################################################################
def sentiment_balanced_triangles(G, weight_attribute='weight'):
    """
    Finds balanced and unbalanced triangles (triads) in a signed graph.
    Balanced if the product of signs is positive, unbalanced if negative.
    Returns counts of balanced/unbalanced plus fraction balanced.
    """
    balanced_count = 0
    unbalanced_count = 0

    # Use cycle_basis on the undirected version for 3-cycles
    for cycle in nx.cycle_basis(G.to_undirected()):
        if len(cycle) == 3:
            n1, n2, n3 = cycle
            if G.has_edge(n1, n2) and G.has_edge(n2, n3) and G.has_edge(n3, n1):
                w12 = G[n1][n2].get(weight_attribute, 1.0)
                w23 = G[n2][n3].get(weight_attribute, 1.0)
                w31 = G[n3][n1].get(weight_attribute, 1.0)

                # Multiply signs
                sign_product = np.sign(w12) * np.sign(w23) * np.sign(w31)
                if sign_product > 0:
                    balanced_count += 1
                else:
                    unbalanced_count += 1

    total_triangles = balanced_count + unbalanced_count
    fraction_balanced = balanced_count / total_triangles if total_triangles else 0.0

    return {
        'balanced_count': balanced_count,
        'unbalanced_count': unbalanced_count,
        'fraction_balanced': fraction_balanced
    }

###############################################################################
#                              CENTRALITIES                                   #
###############################################################################
def compute_centralities(G):
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    return betweenness, closeness

###############################################################################
#                           TOP 10 CENTRALITIES / DEGREE                      #
###############################################################################
def list_top_10_centralities(betweenness, closeness):
    betw_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    clos_sorted = sorted(closeness.items(), key=lambda x: x[1], reverse=True)

    betw_rank = {}
    for i, (node, val) in enumerate(betw_sorted):
        betw_rank[node] = i + 1

    clos_rank = {}
    for i, (node, val) in enumerate(clos_sorted):
        clos_rank[node] = i + 1

    top_10_betw = betw_sorted[:10]
    top_10_clos = clos_sorted[:10]

    data_betw = []
    data_clos = []
    for node, val in top_10_betw:
        data_betw.append((node, val, clos_rank[node]))
    for node, val in top_10_clos:
        data_clos.append((node, val, betw_rank[node]))

    return data_betw, data_clos

def get_top_degree_nodes(G, n=10):
    """
    Returns the top-n nodes by degree (descending).
    """
    sorted_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return sorted_by_degree[:n]

###############################################################################
#                           COMMUNITY DETECTION (LOUVAIN)                     #
###############################################################################
def run_louvain_with_abs_weights(G):
    """
    Convert negative weights to absolute values, then run Louvain.
    Returns the partition dict {node -> commID}.
    """
    full_graph_abs = copy.deepcopy(G)
    for u, v in full_graph_abs.edges():
        w = full_graph_abs[u][v].get('weight', 0)
        if w < 0:
            full_graph_abs[u][v]['weight'] = abs(w)
    partition = community_louvain.best_partition(full_graph_abs)
    return partition

###############################################################################
#                    INTERNAL SENTIMENT (FOR COMMUNITIES)                     #
###############################################################################
def new_cluster_sentiment(G, partition_dict):
    """
    Compute average internal edge weight for each Louvain community using G's edges.
    Returns a list of average sentiment values, one per community index.
    """
    comm_map = defaultdict(set)
    for node, cid in partition_dict.items():
        comm_map[cid].add(node)
    communities = list(comm_map.values())

    internal_sent = []
    for cset in communities:
        sub_edges = G.subgraph(cset).edges(data='weight', default=0)
        weights = [d for (_, _, d) in sub_edges]
        if weights:
            internal_sent.append(np.mean(weights))
        else:
            internal_sent.append(0.0)

    return internal_sent

###############################################################################
#            PLOT AVERAGE INTERNAL SENTIMENT FOR COMMUNITIES (BAR CHART)      #
###############################################################################
def plot_communities_internal_sentiment(internal_sent, out_png):
    """
    Plots the average internal sentiment of each community as a bar chart.
    """
    plt.figure(figsize=(8, 5))
    xvals = range(len(internal_sent))
    plt.bar(xvals, internal_sent, color='skyblue')
    plt.xlabel("Community Index")
    plt.ylabel("Average Internal Sentiment")
    plt.title("Average Internal Sentiment (Louvain Communities, Full Graph)")
    plt.xticks(xvals, [str(i) for i in xvals])
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

###############################################################################
#                             DEGREE ASSORTATIVITY                            #
###############################################################################
def degree_assortativity(G):
    """Returns the Pearson correlation coefficient for node degrees."""
    return nx.degree_pearson_correlation_coefficient(G)

###############################################################################
#                            POWER-LAW DISTRIBUTION                            #
###############################################################################
def plot_save_degree_distribution(G, title="Degree Distribution", filename=None):
    """
    Plots and saves the degree distribution on a log-log scale.
    Also estimates and returns the power-law exponent via linear regression.
    """
    degrees = dict(G.degree())
    values = list(degrees.values())
    unique_degs = sorted(set(values))
    count_of_degs = [values.count(x) for x in unique_degs]
    total_nodes = G.number_of_nodes()
    pk = [c / total_nodes for c in count_of_degs]

    # Plot
    plt.figure()
    plt.scatter(unique_degs, pk)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Degree (log)')
    plt.ylabel('P(k) (log)')

    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()

    # Estimate power-law exponent
    power_law_exponent = None
    x_vals, y_vals = [], []
    for k, p in zip(unique_degs, pk):
        if k > 1 and p > 0:
            x_vals.append(math.log(k))
            y_vals.append(math.log(p))

    if len(x_vals) > 2:
        A = np.vstack([x_vals, np.ones(len(x_vals))]).T
        slope, intercept = np.linalg.lstsq(A, y_vals, rcond=None)[0]
        power_law_exponent = -slope

    return power_law_exponent

###############################################################################
#                                  EDGES                                      #
###############################################################################
def get_top_10_edges(graph, output_file=None):
    """
    Collect and (optionally) store the top 10 edges with the highest weight.
    Returns the list of top-10 edges as (node1, node2, weight).
    """
    edges_sorted = sorted(
        graph.edges(data=True), 
        key=lambda x: x[2].get('weight', 0), 
        reverse=True
    )
    top_10_edges = edges_sorted[:10]

    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Node1', 'Node2', 'Weight'])
            for u, v, data in top_10_edges:
                weight = data.get('weight', 0)
                writer.writerow([u, v, weight])

    return top_10_edges

###############################################################################
#         HELPER FUNCTION: SAVE TOP-10 EDGES/DEGREE/BETW/CLOS FOR A GRAPH     #
###############################################################################
def save_top_10_metrics(graph, graph_name):
    """
    Saves the top 10 edges (with negative->abs if full),
    top 10 degree, top 10 betweenness, and top 10 closeness
    to CSV files in the OUTPUT_DIR, named with the graph_name prefix.
    """
    ensure_output_dir()

    # 1) Top 10 edges
    if graph_name == "full":
        # Convert negative weights to abs, then get top edges
        full_graph_abs = copy.deepcopy(graph)
        for u, v in full_graph_abs.edges():
            w = full_graph_abs[u][v].get('weight', 0)
            if w < 0:
                full_graph_abs[u][v]['weight'] = abs(w)

        edges_csv = os.path.join(OUTPUT_DIR, f"top10_edges_{graph_name}.csv")
        get_top_10_edges(full_graph_abs, output_file=edges_csv)

    else:
        edges_csv = os.path.join(OUTPUT_DIR, f"top10_edges_{graph_name}.csv")
        get_top_10_edges(graph, output_file=edges_csv)

    # 2) Top 10 degree
    degree_csv = os.path.join(OUTPUT_DIR, f"top10_degree_{graph_name}.csv")
    sorted_by_degree = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    top_degree = sorted_by_degree[:10]
    with open(degree_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'Degree'])
        for node, deg in top_degree:
            writer.writerow([node, deg])

    # 3) Betweenness and closeness
    betw, clos = compute_centralities(graph)

    betw_csv = os.path.join(OUTPUT_DIR, f"top10_betweenness_{graph_name}.csv")
    sorted_betw = sorted(betw.items(), key=lambda x: x[1], reverse=True)[:10]
    with open(betw_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'Betweenness'])
        for node, bval in sorted_betw:
            writer.writerow([node, bval])

    clos_csv = os.path.join(OUTPUT_DIR, f"top10_closeness_{graph_name}.csv")
    sorted_clos = sorted(clos.items(), key=lambda x: x[1], reverse=True)[:10]
    with open(clos_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'Closeness'])
        for node, cval in sorted_clos:
            writer.writerow([node, cval])

###############################################################################
#                                   RANDOM GRAPHS                             #
###############################################################################

def generate_random_graph_equivalent(G):
    """
    Generates a random Erdős–Rényi graph (G(n, p)) with the same number of nodes
    and roughly the same density as the given graph G.
    
    Parameters:
        G (networkx.Graph): The original graph.
    
    Returns:
        (networkx.Graph): A random Erdős–Rényi graph with:
            - n = G.number_of_nodes()
            - p = (E / [n*(n-1)/2]), where E is the number of edges in G.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    
    # If n < 2, we can't define a probability meaningfully.
    if n < 2:
        return nx.empty_graph(n)
    
    # Calculate approximate density
    max_edges = n*(n-1)/2
    p = e / max_edges if max_edges > 0 else 0
    
    # Generate the random graph with the same number of nodes & approximate p
    random_graph = nx.erdos_renyi_graph(n, p)
    return random_graph

###############################################################################
#                                   CLUSTERING COEFFICIENT                    #
###############################################################################

def calculate_average_clustering_coefficient(G):
    """
    Calculates the average clustering coefficient of a graph.

    Parameters:
        G (networkx.Graph): Input graph.

    Returns:
        avg_clustering (float): Average clustering coefficient.
    """
    # Use networkx to calculate the average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    return avg_clustering

###############################################################################
#                              K-CORE SENTIMENT                               #
###############################################################################

def collect_kcore_sentiment_distribution(graph, output_file, sentiment_attr='weight'):
    """
    Collect and store sentiment distributions across different k-core levels.
    """
    # Get the k-core numbers for all nodes
    core_dict = nx.core_number(graph)

    # Group nodes by their k-core level
    kcore_groups = defaultdict(list)
    for node, core in core_dict.items():
        kcore_groups[core].append(node)

    # Collect sentiment data for each k-core level
    results = []
    for kcore, nodes in kcore_groups.items():
        subgraph = graph.subgraph(nodes)
        sentiments = [
            data[sentiment_attr]
            for _, _, data in subgraph.edges(data=True)
            if sentiment_attr in data
        ]
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_std = (sum((x - avg_sentiment) ** 2 for x in sentiments) / len(sentiments)) ** 0.5
        else:
            avg_sentiment = 0
            sentiment_std = 0

        results.append((kcore, avg_sentiment, sentiment_std, len(sentiments)))

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['K-Core Level', 'Average Sentiment', 'Sentiment Std Dev', 'Number of Edges'])
        writer.writerows(results)

###############################################################################
#                                   MAIN                                      #
###############################################################################
def main():
    """
    Runs all analyses on the full, positive, negative graphs; logs results;
    performs Louvain on the full graph with absolute weights and plots internal sentiment.
    """
    ensure_output_dir()

    # Open a text file to store detailed logs
    log_file_path = os.path.join(OUTPUT_DIR, "analysis_log.txt")
    with open(log_file_path, 'w', encoding='utf-8') as out_log:
        
        def log_write(*args):
            """Write text to 'analysis_log.txt' (not to the console)."""
            out_log.write(" ".join(str(a) for a in args) + "\n")

        # Minimal terminal output
        print("Loading CSV-based graphs...")

        #-----------------------------------------------------------------------
        # 1) LOAD GRAPHS
        #-----------------------------------------------------------------------
        full_graph = load_full_graph_from_csv("topic_graph_csv/full.csv")
        positive_graph = load_signed_graph_from_csv("topic_graph_csv/positive.csv")
        negative_graph = load_signed_graph_from_csv("topic_graph_csv/negative.csv")
        
        # Full graph
        G_full, edge_data_full = process_all_json_files(CATEGORY_FOLDER)
        save_graph_to_gexf(G_full, OUTPUT_GEXF_FULL)

        # Positive-only graph
        G_positive, edge_data_positive = process_all_json_files(CATEGORY_FOLDER, positive_only=True)
        save_graph_to_gexf(G_positive, OUTPUT_GEXF_POS)

        # Negative-only graph
        G_negative, edge_data_negative = process_all_json_files(CATEGORY_FOLDER, negative_only=True)
        save_graph_to_gexf(G_negative, OUTPUT_GEXF_NEG)

        log_write("==== [1] NUMBER OF NODES IN EACH GRAPH ====")
        log_write("Full graph:", full_graph.number_of_nodes())
        log_write("Positive graph:", positive_graph.number_of_nodes())
        log_write("Negative graph:", negative_graph.number_of_nodes())

        print("Counting unique topics (JSON files)...")
        #-----------------------------------------------------------------------
        # 2) COUNT UNIQUE TOPICS
        #-----------------------------------------------------------------------
        folder = "articles_by_year"  # adjust if needed
        unique_count, _ = count_unique_topics(folder)
        log_write("\n==== [2] NUMBER OF UNIQUE TOPICS ====")
        log_write("Unique topics count:", unique_count)

        print("Calculating balanced/unbalanced triangles...")
        #-----------------------------------------------------------------------
        # 3) BALANCED & UNBALANCED TRIADS (FULL GRAPH)
        #-----------------------------------------------------------------------
        triads_info = sentiment_balanced_triangles(full_graph)
        log_write("\n==== [3] TRIADS (FULL GRAPH) ====")
        log_write("Balanced count:", triads_info['balanced_count'])
        log_write("Unbalanced count:", triads_info['unbalanced_count'])
        log_write("Fraction balanced:", triads_info['fraction_balanced'])

        print("Plotting degree distributions & computing power-law exponent...")
        #-----------------------------------------------------------------------
        # 4) POWER LAW & DEGREE DISTRIBUTION
        #-----------------------------------------------------------------------
        exponent_full = plot_save_degree_distribution(
            full_graph,
            title="Full Network Degree Dist",
            filename=os.path.join(OUTPUT_DIR, "degree_dist_full.png")
        )
        exponent_pos = plot_save_degree_distribution(
            positive_graph,
            title="Positive-only Degree Dist",
            filename=os.path.join(OUTPUT_DIR, "degree_dist_pos.png")
        )
        exponent_neg = plot_save_degree_distribution(
            negative_graph,
            title="Negative-only Degree Dist",
            filename=os.path.join(OUTPUT_DIR, "degree_dist_neg.png")
        )

        log_write("\n==== [4] POWER-LAW EXPONENT ESTIMATES ====")
        log_write("Full graph exponent:", exponent_full)
        log_write("Positive graph exponent:", exponent_pos)
        log_write("Negative graph exponent:", exponent_neg)

        print("Calculating degree assortativity...")
        #-----------------------------------------------------------------------
        # 5) DEGREE ASSORTATIVITY
        #-----------------------------------------------------------------------
        assort_full = degree_assortativity(full_graph)
        assort_pos = degree_assortativity(positive_graph)
        assort_neg = degree_assortativity(negative_graph)

        log_write("\n==== [5] DEGREE ASSORTATIVITY ====")
        log_write("Full graph assortativity:", assort_full)
        log_write("Positive graph assortativity:", assort_pos)
        log_write("Negative graph assortativity:", assort_neg)

        print("Computing betweenness & closeness (averages only in log) ...")
        #-----------------------------------------------------------------------
        # 6) AVERAGE BETWENNESS & CLOSENESS
        #-----------------------------------------------------------------------
        betw_full, clos_full = compute_centralities(full_graph)
        avg_betw_full = np.mean(list(betw_full.values()))
        avg_clos_full = np.mean(list(clos_full.values()))

        betw_pos, clos_pos = compute_centralities(positive_graph)
        avg_betw_pos = np.mean(list(betw_pos.values()))
        avg_clos_pos = np.mean(list(clos_pos.values()))

        betw_neg, clos_neg = compute_centralities(negative_graph)
        avg_betw_neg = np.mean(list(betw_neg.values()))
        avg_clos_neg = np.mean(list(clos_neg.values()))

        log_write("\n==== [6] AVERAGE BETWENNESS & CLOSENESS ====")
        log_write("Full -> avg betweenness:", avg_betw_full, "avg closeness:", avg_clos_full)
        log_write("Positive -> avg betweenness:", avg_betw_pos, "avg closeness:", avg_clos_pos)
        log_write("Negative -> avg betweenness:", avg_betw_neg, "avg closeness:", avg_clos_neg)

        print("Saving top 10 edges, degree, betweenness, closeness for all three graphs...")
        #-----------------------------------------------------------------------
        # 7) SAVE TOP-10 METRICS (EDGES, DEGREE, BETW/CLOS) FOR EACH GRAPH
        #-----------------------------------------------------------------------
        save_top_10_metrics(full_graph, "full")
        save_top_10_metrics(positive_graph, "positive")
        save_top_10_metrics(negative_graph, "negative")

        log_write("\n==== [7] TOP-10 METRICS FOR ALL GRAPHS SAVED ====")

        print("Running Louvain on the Full graph with abs weights & plotting internal sentiment...")
        #-----------------------------------------------------------------------
        # 8) LOUVAIN (ABS WEIGHTS) + INTERNAL SENTIMENT PLOT
        #-----------------------------------------------------------------------
        louvain_part = run_louvain_with_abs_weights(full_graph)
        int_sent = new_cluster_sentiment(full_graph, louvain_part)
        out_sent_plot = os.path.join(OUTPUT_DIR, "louvain_communities_internal_sentiment_full.png")
        plot_communities_internal_sentiment(int_sent, out_sent_plot)

        log_write("\n==== [8] LOUVAIN COMMUNITIES (ABS WEIGHTS, FULL GRAPH) ====")
        log_write(f"Number of communities found: {len(set(louvain_part.values()))}")
        log_write("Average internal sentiments:", int_sent)
        log_write("(A bar chart was saved to 'louvain_communities_internal_sentiment_full.png')")
        
        
        # 9) CLUSTERING COEFFICIENT
        
        cc_full = calculate_average_clustering_coefficient(full_graph)
        cc_pos = calculate_average_clustering_coefficient(positive_graph)
        cc_neg = calculate_average_clustering_coefficient(negative_graph)
        log_write("\n==== [9] CLUSTERING COEFFICIENTs ====")
        log_write(f"Full graph: {cc_full}")
        log_write(f"Positive graph: {cc_pos}")
        log_write(f"Negative graph: {cc_neg}")
        
        # 10) RANDOM GRAPHS AND CLUSTERING COEFFICIENT
        full_random = generate_random_graph_equivalent(full_graph)
        pos_random = generate_random_graph_equivalent(positive_graph)
        neg_random = generate_random_graph_equivalent(negative_graph)
        cc_full_random = calculate_average_clustering_coefficient(full_random)
        cc_pos_random = calculate_average_clustering_coefficient(pos_random)
        cc_neg_random = calculate_average_clustering_coefficient(neg_random)
        log_write("\n==== [10] CLUSTERING COEFFICIENTs FOR RANDOM GRAPH ====")
        log_write(f"Full graph: {cc_full_random}")
        log_write(f"Positive graph: {cc_pos_random}")
        log_write(f"Negative graph: {cc_neg_random}")
        
        # 11) K-CORE SENTIMENT
        collect_kcore_sentiment_distribution(full_graph, "analysed_topic_data/kcore_sentiment_full.csv")
        collect_kcore_sentiment_distribution(positive_graph, "analysed_topic_data/kcore_sentiment_pos.csv")
        collect_kcore_sentiment_distribution(negative_graph,  "analysed_topic_data/kcore_sentiment_neg.csv")
        
        log_write("\nAnalysis complete.")
        print("Analysis complete. See 'analysis_log.txt' for details.\n")


# Standard entry point
if __name__ == "__main__":
    main()
