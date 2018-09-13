import matplotlib.pyplot as plt
import plotly
plotly.tools.set_credentials_file(username='TODO', api_key = 'TODO')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')

import sys
sys.path.insert(0, '/home/jovyan/')

from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.metrics import mutual_info_score

import ccal.ccal as ccal
from msigdb_decomp.SNMF.master_SNMF import SNMF as SimulNMF

from numpy.linalg import LinAlgError

FONTSIZE = 24

def gct_process(data, genes, data_type):
    """
    Preprocesses a .gct file using genes from file specified by type

        data (string): path to .gct file containing gene expression data
        genes: specification of which gene set to use
        data_type (string) : keyword indicating what type of genes inputted.
                            Options are:
                                "txt" : string to path for .txt file
                                "lst" : [string] of gene names

        returns (data frame): processed gene expression data
        returns : index by gene set
        returns : columns by sample name
    """
    TYPE_ERROR = "data_type error: Options are: txt; lst"
    ## load the data
    raw_data = ccal.read_gct(data)
    print(raw_data.shape)
    ## load the genes
    if (data_type == 'txt'):
        SIG = sorted(open(genes).read().split('\n'))
    elif (data_type == 'lst'):
        SIG = genes
    else:
        print(TYPE_ERROR)
    ## keep only the genes
    raw_data = raw_data.loc[SIG,:].dropna(axis=0, how='all')
    print(raw_data.shape)
    ## Sort by genes 
    raw_data = raw_data.groupby(raw_data.index, axis=0).max()
    ## Convert to dense ranks
    raw_data = raw_data.rank(axis=0, method='dense', numeric_only=None, na_option='bottom',
                    ascending=True, pct=False)
    ## scale between 1, 10,000
    raw_data = 10000*(raw_data-raw_data.min())/(raw_data.max() - raw_data.min())
    gene_index = raw_data.index
    sample_cols = raw_data.columns
    return (raw_data, gene_index, sample_cols)

def assign_clust(H):
    """
    This function takes an H matrix and
    returns a list of cluster assignment for each sample. 
    Samples are assigned a cluster by taking the maximum of each column.

        H (np.matrix): H matrix from an NMF decomposition

        returns (list): cluster assignment of each sample
    """
    output = [col.index(max(col)) for col in H.T.tolist()]
    cluster_breakdown = [output.count(i) for i in range(H.shape[0])]
    print(cluster_breakdown)
    return (output, cluster_breakdown)

def get_conn_mat(clusts):
    """
    This function takes an H matrix and
    returns a list of cluster assignment for each sample. 
    are assigned a cluster by taking the maximum of each column.

        clusts (list): list indicating sample assingment

        returns (np.matrix): binary connectivity matrix
    """
    dim = len(clusts)
    conn_mat = np.matrix(np.zeros((dim, dim)))
    for i in range(dim):
        for j in range(dim):
            if clusts[i] == clusts[j]:
                conn_mat[i,j] = 1
    return conn_mat

def get_avg_con_mats(A, k, n_iter, ALG, START):
    """
    Returns the average connectivity matrix based on NMF specified by ALG

        A : data-set to decompose
        k (int): number of metagenes
        n_iter (int): number of different decompositionn to average
        ALG (string) : Which update to perform. 
                                options are: 
                                'base' : (simultaneous NMF) 
                                'sorth_W' : (simultaneous NMF with semi-orthogonal W)
                                'sorth_H : (simultaneous NMF with semi-orthogonal H)
                                'norm_sorth_W' : (simultaneous NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'norm_sorth_H' : (simultaneous NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
                                'aff_sorth_W' : (simultaneous affine NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'aff_sorth_H' : (simultaneous affine NMF with semi-orthogonal H
                                                where rows of H normalized every iteration) 
       START (string): How to initialize matrices.
                           options are:
                           'rand' : random initialization
                           'sorth_W' : semi-orthogonal W
                           'sorth_H' : semi-orthogonal H

        returns (np.matrix): Average Connectivity matrix
    """
    print('starting to get average connectvity matrix')
    Cb = np.matrix(np.zeros((A.shape[1], A.shape[1])))
    for i in range(n_iter):
        snmf = SimulNMF([A], n_comps=k, n_iter=2000, resid_thresh=1e-15, alg = ALG, start = START)
        Cb += get_conn_mat(assign_clust(snmf.Hs[0])[0])
    Cb = Cb/n_iter
    print()
    print("the sum of entries is " + str(Cb.sum()))
    print()
    return Cb
    #return sns.clustermap(Cb, cmap = 'bwr',  vmin = 0, vmax = 1)

def get_cophenetic_scipy(A, k, n_iter, alg, start):
    """
    Returns the cophenetic correlation coefficient for NMF (specified by alg) with k metagenes

        A : data-set to decompose
        k (int): number of metagenes
        n_iter (int): number of different decompositionn to average
        alg (string) : Which variant of SNMF to perform. 
                                options are: 
                                'base' : (simultaneous NMF) 
                                'sorth_W' : (simultaneous NMF with semi-orthogonal W)
                                'sorth_H : (simultaneous NMF with semi-orthogonal H)
                                'norm_sorth_W' : (simultaneous NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'norm_sorth_H' : (simultaneous NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
                                'aff_sorth_W' : (simultaneous affine NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'aff_sorth_H' : (simultaneous affine NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
       start (string): How to initialize matrices.
                           options are:
                           'rand' : random initialization
                           'sorth_W' : semi-orthogonal W
                           'sorth_H' : semi-orthogonal H
                           
        returns (float): cophenetic correlation coefficient for simultaeneous NMF with k metagenes
     """
    Cb = get_avg_con_mats(A, k, n_iter, alg,start)
    lmat = linkage(Cb, method='average')
    return cophenet(lmat, pdist(Cb))[0]

def coph_lst(data, ks, n_iter, alg, start):
    """
    Returns the cophenetic correlation coefficients for different values of k

        data : data-set to decompose
        ks (list): values of k (number of metagenes) over which to range
        n_iter (int): number of different decompositionn to average
        alg (string) : Which variant of SNMF to perform. 
                                options are: 
                                'base' : (simultaneous NMF) 
                                'sorth_W' : (simultaneous NMF with semi-orthogonal W)
                                'sorth_H : (simultaneous NMF with semi-orthogonal H)
                                'norm_sorth_W' : (simultaneous NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'norm_sorth_H' : (simultaneous NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
                                'aff_sorth_W' : (simultaneous affine NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'aff_sorth_H' : (simultaneous affine NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
       start (string): How to initialize matrices.
                           options are:
                           'rand' : random initialization
                           'sorth_W' : semi-orthogonal W
                           'sorth_H' : semi-orthogonal H

        returns ([float]): list of cophenetic coefficients for different values of k
     """
    return [get_cophenetic_scipy(data, k, n_iter, alg, start) for k in ks]
    

def get_pairwise_IC(w1, w2):
    """
    Gives pairwise IC matrix between columns of two W matrices

        w1 (data frame): first W matrix
        w2 (data frame): second W matrix

        returns (np.matrix): pairwise IC matrix
    """
    ic_mat = np.empty((w1.shape[1], w2.shape[1]))
    for i in range(w1.shape[1]):
        for j in range(w2.shape[1]):
            ic_mat[i,j] = ccal.compute_information_coefficient(
                np.asarray(w1.iloc[:,i]),
                np.asarray(w2.iloc[:,j])
        )
    return ic_mat

def get_pairwise_row_IC(h1, h2):
    """
    Gives pairwise IC matrix between rows of two H matrices

        h1 (data frame): first H matrix
        h2 (data frame): second H matrix

        returns (np.matrix): pairwise IC matrix
    """
    ic_mat = np.empty((h1.shape[0], h2.shape[0]))
    if h1.shape[1] == h2.shape[1]:
        for i in range(h1.shape[0]):
            for j in range(h2.shape[0]):
                ic_mat[i,j] = ccal.compute_information_coefficient(
                    np.asarray(h1.iloc[i,:]),
                    np.asarray(h2.iloc[j,:])
            )
    elif h1.shape[1] < h2.shape[1]:
        diff = h2.shape[1] - h1.shape[1]
        for i in range(h1.shape[0]):
            for j in range(h2.shape[0]):
                ic_mat[i,j] = ccal.compute_information_coefficient(
                    np.asarray(h1.iloc[i,:]),
                    np.asarray(h2.iloc[j,:-diff])
            )
    elif h1.shape[1] > h2.shape[1]:
        diff = h1.shape[1] - h2.shape[1]
        for i in range(h1.shape[0]):
            for j in range(h2.shape[0]):
                ic_mat[i,j] = ccal.compute_information_coefficient(
                    np.asarray(h1.iloc[i,:-diff]),
                    np.asarray(h2.iloc[j,:])
            )
    return ic_mat

def simul_eval(data_set, INDEX, COLS, numb_comps, n_iter, ks, resid_threshold, ALG, START):
    """
    Plots pairwise IC matrices for Ws and Hs resulting 
    from multiple runs of NMF

        data_set (data frame): gene expression data set
        INDEX : index for genes
        COLS : index for samples
        numb_comps (int) : number of metagenes
        n_iter (int) : number of NMFs to run
        ks (list): values of k (number of metagenes) over which to range
        resid_threshold =  Threshold value for the ratio of the 
                           difference in recent errors to the original
                           residual of the factored matrix.
        ALG (string) : Which update of SNMF to use. 
                                options are: 
                                'base' : (simultaneous NMF) 
                                'sorth_W' : (simultaneous NMF with semi-orthogonal W)
                                'sorth_H : (simultaneous NMF with semi-orthogonal H)
                                'norm_sorth_W' : (simultaneous NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'norm_sorth_H' : (simultaneous NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
                                'aff_sorth_W' : (simultaneous affine NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'aff_sorth_H' : (simultaneous affine NMF with semi-orthogonal H
                                                where rows of H normalized every iteration) 
       START (string): How to initialize matrices.
                           options are:
                           'rand' : random initialization
                           'sorth_W' : semi-orthogonal W
                           'sorth_H' : semi-orthogonal H
    """
    print("Average Connectivity Matrix")
    print()
    sns.clustermap(get_avg_con_mats(data_set, numb_comps, n_iter, ALG, START), cmap = 'bwr',  vmin = 0, vmax = 1)
    plt.show()
    print()
    #print("The ks are " + str(ks))
    try:
        coph_coefs = coph_lst(data_set, ks, n_iter, ALG, START)
        #print("The cophenetic coefficients are " + str(coph_coefs))
        if (len(ks) > 1):
            print("Plot of Cophenetic Correlation Coefficients")
            plt.plot(ks, coph_coefs)
            plt.xlabel('k')
            plt.ylabel('coph coeff')
            plt.ylim(np.nanmin(coph_coefs) - 0.1, 1)
            plt.title(str(ALG) + " update; " + str(START) + " start " + ' cophenetic coefficient vs. k')
            plt.show()
    except LinAlgError:
        print("cophenetic coefficients could not compute")
    NMF_1 = SimulNMF([data_set], n_comps=numb_comps, n_iter=2000, resid_thresh = resid_threshold, alg = ALG, start = START)
#     print('1 NMF')
    NMF_2 = SimulNMF([data_set], n_comps=numb_comps, n_iter=2000, resid_thresh = resid_threshold, alg = ALG, start = START)
#     print('2 NMF')
    NMF_3 = SimulNMF([data_set], n_comps=numb_comps, n_iter=2000, resid_thresh = resid_threshold, alg = ALG, start = START)
#     print('3 NMF')
    NMF_4 = SimulNMF([data_set], n_comps=numb_comps, n_iter=2000, resid_thresh = resid_threshold, alg = ALG, start = START)
#     print('4 NMF')
    W_1 = pd.DataFrame(NMF_1.W, index = INDEX)
    W_2 = pd.DataFrame(NMF_2.W, index = INDEX)
    W_3 = pd.DataFrame(NMF_3.W, index = INDEX)
    W_4 = pd.DataFrame(NMF_4.W, index = INDEX)
    mats = [W_1,W_2, W_3, W_4]
    
    VMIN = min(NMF_1.W.min(), NMF_2.W.min(),NMF_3.W.min(),NMF_3.W.min())
    VMAX = max(NMF_1.W.max(), NMF_2.W.max(),NMF_3.W.max(),NMF_3.W.max())
    
    print("Starting orthogonality of W matrices")
    print(NMF_1.W_orth[0])
    print(NMF_2.W_orth[0])
    print(NMF_3.W_orth[0])
    print(NMF_4.W_orth[0])
    print()
    
    print("Ending orthogonality of W matrices")
    print(NMF_1.W_orth[-1])
    print(NMF_2.W_orth[-1])
    print(NMF_3.W_orth[-1])
    print(NMF_4.W_orth[-1])
    print()
    
    print("Heatmaps of W matrices")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize = (20,10))
    for i in range(len(mats)):
        sns.heatmap(mats[i], ax = axes[i], cmap = 'bwr', vmin = VMIN, vmax = VMAX)
        
    axes[0].set_title("W 1", fontsize=FONTSIZE)
    axes[1].set_title("W 2", fontsize=FONTSIZE)
    axes[2].set_title("W 3", fontsize=FONTSIZE)
    axes[3].set_title("W 4", fontsize=FONTSIZE)
    
    plt.show()
    
    print('Pairwise information coefficients between columns of W')
    print()
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15,12))
    for i in range(len(mats)):
        for j in range(len(mats)):
            sns.heatmap(get_pairwise_IC(mats[i], mats[j]), ax=axes[i,j], cmap='bwr', vmin=-1, vmax=1)

    axes[0,0].set_title("W 1", fontsize=FONTSIZE)
    axes[0,1].set_title("W 2", fontsize=FONTSIZE)
    axes[0,2].set_title("W 3", fontsize=FONTSIZE)
    axes[0,3].set_title("W 4", fontsize=FONTSIZE)

    axes[0,0].set_ylabel("W 1", fontsize=FONTSIZE)
    axes[1,0].set_ylabel("W 2", fontsize=FONTSIZE)
    axes[2,0].set_ylabel("W 3", fontsize=FONTSIZE)
    axes[3,0].set_ylabel("W 4", fontsize=FONTSIZE)

    plt.show()
    
    H_1 = pd.DataFrame(NMF_1.Hs[0], columns = COLS)
    H_2 = pd.DataFrame(NMF_2.Hs[0], columns = COLS)
    H_3 = pd.DataFrame(NMF_3.Hs[0], columns = COLS)
    H_4 = pd.DataFrame(NMF_4.Hs[0], columns = COLS)
    mats = [H_1,H_2, H_3, H_4]
    
    VMIN = min(NMF_1.Hs[0].min(), NMF_2.Hs[0].min(),NMF_3.Hs[0].min(),NMF_3.Hs[0].min())
    VMAX = max(NMF_1.Hs[0].max(), NMF_2.Hs[0].max(),NMF_3.Hs[0].max(),NMF_3.Hs[0].max())
    
    print("Starting orthogonality of H matrices")
    print(NMF_1.Hs_orth[0])
    print(NMF_2.Hs_orth[0])
    print(NMF_3.Hs_orth[0])
    print(NMF_4.Hs_orth[0])
    print()
    
    print("Ending orthogonality of H matrices")
    print(NMF_1.Hs_orth[-1])
    print(NMF_2.Hs_orth[-1])
    print(NMF_3.Hs_orth[-1])
    print(NMF_4.Hs_orth[-1])
    print()
    
    print("Heatmaps of H matrices")
    fig, axes = plt.subplots(nrows=1, ncols=4,figsize = (25,5))
    for i in range(len(mats)):
        sns.heatmap(mats[i], ax = axes[i], cmap = 'bwr', vmin = VMIN, vmax = VMAX)
        
    axes[0].set_title("H 1", fontsize=FONTSIZE)
    axes[1].set_title("H 2", fontsize=FONTSIZE)
    axes[2].set_title("H 3", fontsize=FONTSIZE)
    axes[3].set_title("H 4", fontsize=FONTSIZE)
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15,12))
    print('Pairwise information coefficients between rows of H')
    print()
    for i in range(len(mats)):
        for j in range(len(mats)):
            sns.heatmap(get_pairwise_row_IC(mats[i], mats[j]), ax=axes[i,j], cmap='bwr', vmin=-1, vmax=1)

    axes[0,0].set_title("H 1", fontsize=FONTSIZE)
    axes[0,1].set_title("H 2", fontsize=FONTSIZE)
    axes[0,2].set_title("H 3", fontsize=FONTSIZE)
    axes[0,3].set_title("H 4", fontsize=FONTSIZE)

    axes[0,0].set_ylabel("H 1", fontsize=FONTSIZE)
    axes[1,0].set_ylabel("H 2", fontsize=FONTSIZE)
    axes[2,0].set_ylabel("H 3", fontsize=FONTSIZE)
    axes[3,0].set_ylabel("H 4", fontsize=FONTSIZE)

    plt.show()
    
def intersect(a, b):
    """
    return the intersection of two lists
    """
    return list(set(a) & set(b))

def repeated_intersection(lst):
    """
    The repeated intersection of a list of lists
    
        lst ([list])
    """
    core = lst[0]
    for l in lst:
        core = intersect(core, l)
    return core

def sankey_nodes(s, t):
    """
    Returns labeling for nodes of Sankey diagram
    
        s (int) : number of sources
        t (int) : number of targets
        
        returns [string] : dictionary needed to specify nodes in Sankey diagram
    """
    COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
    LABEL = [str(s) + '-' + str(i) for i in range(s)] +  [str(t) + '-' + str(i) for i in range(t)]
    COLOR = [COLORS[i] for i in range(s)] + [COLORS[i] for i in range(t)] 
    return dict(
            label = LABEL,
            color = COLOR)

def sankey_links(s,t):
    """
    Returns dictionary needed to specify links in Sankey diagram
    
        s (dict) : dictionary of source decomposition
        t (dict) : dictionary of target decomposition
        
        returns (dict) : dictionary needed to specify links in Sankey diagram
    """
    s_length = len(s)
    t_length = len(t)
    SOURCE = []
    for i in range(t_length):
        strts = [j for j in range(s_length)]
        SOURCE += strts
    TARGET = []
    for i in range(t_length):
        strts = [i + s_length for j in range(s_length)]
        TARGET += strts
    return dict(
        source = SOURCE, 
        target = TARGET, 
        value = [len(intersect(s[i][0], t[j][0])) for j in range(t_length) for i in range(s_length)])

def sankey_diagram(s,t):
    """
    Genertes a Sankey flow digram between two different decompositions
    
        s (dict) : dictionary giving source decomposition
        t (dict) : dictionary giving target decomposition
        
        plots appropriate Sankey diagram
    """
    data = dict(
    type='sankey',
    node = dict(
      label = sankey_nodes(len(s), len(t)),
    ),
    link = sankey_links(s, t)
    )

    layout =  dict(
        title = "Basic Sankey Diagram",
        font = dict(
          size = 10
        )
    )

    fig = dict(data=[data], layout=layout)
    plotly.plotly.iplot(fig, validate=False)
    plotly.plotly.show()
    
"""
gene set code
"""

def As_have_gene(As, gene):
    """
    Detects whether a gene is present in a given collection of gene expression data sets
        
        As ([dataframes]) : list of gene expression pd dataframes
        gene (string) : identifier for gene
        
        returns (bool): True or false depending on whether gene in all dataframes
    """
    for A in As:
        if gene not in A.index:
            return False
    return True

def get_all_genes(As):
    """
    Returns set of all genes in list of gene expression data sets
        
        As ([dataframe]) : list of gene expression pd dataframes
        
        returns ([string]) : list of all genes in these dataframes
    """
    genes = np.array([A.index for A in As]).flatten()
    return list(set(genes))

def get_comp_ICs(As, Hrow):
    """
    Calculates the information coefficient between the expression of a gene 
    and the expression of a metagene
    
        As ([dataframe]): list of gene expression pd dataframes
        Hrow (np.array) : metagene expression
        
        returns (pd.Series) : list of the ICs between the expression of a given gene
                                in all the dataframes and the expression of a metagene
    """
    ics = []
    genes = get_all_genes(As)
    for gene in genes:
        if As_have_gene(As, gene):
            expr = []
            for A in As:
                expr += list(A.loc[gene,:])
            ics.append(ccal.compute_information_coefficient(Hrow, np.array(expr)))
    return pd.Series(ics, index=genes)

def extract_gene_sets(A,H,cut):
    """
    ONLY WORKS FOR SINGLE A MATRIX
    Clusters genes based on similarity to metagene expression
    
        A (dataframe) : gene expression pd dataframe
        H (dataframe) : metagene expression pd dataframe
        cut (float) : threshold IC for adding gene to a metagene cluster
        
        returns (dict) : metagene number : gene set
    """
    k = len(H)
    ICs = [get_comp_ICs([A], H.loc[i,:]) for i in range(k)]
    comps = {}
    for i in range(k):
        comps[i] = list(ICs[i][ICs[i] > cut].index)
    return comps

def create_clusters(cluster_list, sample_columns, clust_numb):
    """
    Returns a dictionary where the keys are cluster numbers and the values are sample members
    
        cluster_list ([int]) : list the length of the number of samples where every entry indicates cluster membership
        sample_columns (index of strings): the names of the samples under consideration
        clust_numb (int) : the number of different clusters available in the decomposition under consideration
        
        returns (dict): keys are cluster numbers, values are sample members
    """
    # include the name of the sample with its cluster assignment
    samples_in_clusters = [(sample_columns[i], cluster_list[i]) for i in range(len(cluster_list))]
    #initializing dictionary:
    sample_clustering = {k: [] for k in range(clust_numb)}
    #dictionary with key: cluster number, value: list of samples
    for t in samples_in_clusters:
        sample_clustering[t[1]].append(t[0])
    return sample_clustering