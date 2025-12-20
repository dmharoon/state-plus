import os
import torch
from scipy.sparse import csr_matrix
import scanpy as sc
import pickle as pkl
import numpy as np
from sklearn.preprocessing import normalize
    
def get_gene_graph_adjacency(gene2go_path, dataset_path):
    with open(gene2go_path, 'rb') as f:
        gene2go = pkl.load(f)
    adata = sc.read_h5ad(dataset_path)
    gene_list = adata.var_names.tolist()
    gene2idx = {gene: i for i, gene in enumerate(gene_list)}
    
    # Rows and columns for a bipartite matrix
    rows = [] # Gene indices
    cols = [] # GO term indices
    
    # Map GO strings to integer IDs
    all_go_terms = set()
    for g in gene_list:
        if g in gene2go:
            all_go_terms.update(gene2go[g])
            
    go2idx = {go: i for i, go in enumerate(sorted(list(all_go_terms)))}
    
    for gene in gene_list:
        if gene in gene2go:
            g_idx = gene2idx[gene]
            for go_term in gene2go[gene]:
                if go_term in go2idx:
                    rows.append(g_idx)
                    cols.append(go2idx[go_term])
    
    # Sparse Bipartite Matrix (Genes x GO Terms)
    data = np.ones(len(rows))
    B = csr_matrix((data, (rows, cols)), shape=(len(gene_list), len(go2idx)))
    
    # Gene-to-Gene Adjacency (Sharing at least one GO term)
    # A[i, j] is the number of GO terms genes i and j share
    A = (B @ B.T)  
    # Remove self-loops
    A.setdiag(0)
    A.eliminate_zeros()
    A_norm = normalize(A, norm='l1', axis=1)
    adj_sparse = A_norm.tocoo()

    adj_dense = adj_sparse.toarray()
    adj_dense_tensor = torch.from_numpy(adj_dense)
    return adj_dense_tensor 

def get_graph_laplacian(adj):
    deg = torch.diag(torch.sum(adj, dim=1))
    laplacian = deg - adj
    return laplacian

def get_genegraph_laplacian():
    # gene2go graph and dataset path for gene list
    gene2go_path="./data/gene2go_all.pkl"
    dataset_path="./competition_support_set/hepg2.h5"
    # Save path : Adjacency of gene to gene graph 
    adjacency_path = "./data/gene2gene_graph_adjacency.pt"
    laplacian_path = "./data/gene2gene_graph_laplacian.pt"
    
    if os.path.exists(adjacency_path):
        adj_dense = torch.load(adjacency_path)    
    else:
        adj_dense = get_gene_graph_adjacency(gene2go_path, dataset_path)
        torch.save(adj_dense, adjacency_path)
    
    # Laplacian of gene to gene graph
    if os.path.exists(laplacian_path):
        graph_laplacian = torch.load(laplacian_path)    
    else:
        graph_laplacian = get_graph_laplacian(adj_dense)
        torch.save(graph_laplacian, laplacian_path)
    return graph_laplacian

