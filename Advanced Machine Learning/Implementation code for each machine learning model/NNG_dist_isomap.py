# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
from sklearn.manifold import MDS, Isomap
import matplotlib.pyplot as plt


def get_knn_graph(X,k):
    # X: input data matrix
    # k: the number of nearest neighbors
    # return knn graph whose element ij is distance between xi and xj if xj is in knn of xi
    
    # knn을 사용해서 각 sample별 k개 만큼 distance graph를 구성함
    # ex) 각 row별 0.....0...0.5....0 = k개
    
    neigh = NearestNeighbors(n_neighbors= k)
    neigh.fit(X)
    
    Kng = neigh.kneighbors_graph(mode='distance')
    #knn_graph = Kng.toarray()
    
    return Kng
    
def cal_dijkstra(knn):
    # knn: knn graph
    # return distance matrix based on knn by Dijkstra algorithm
    
    # pair-wise distance
    dist_matrix =  csgraph.dijkstra(csgraph=knn, directed=False)
    
    return dist_matrix

def isomap(X,k,n_dimension):
    # X: input data matrix
    # k: the number of nearest neighbors
    # n_dimension: reduced dimensionality
    # return low dimensional coordinates 
    # (use get_knn_graph, cal_dijkstra)
    
    knn = get_knn_graph(X,k)
    
    dist_matrix = cal_dijkstra(knn)
    
    MDS_dis_pre = MDS(n_components=n_dimension, dissimilarity='precomputed')
    
    dis_coordn = MDS_dis_pre.fit_transform(dist_matrix)
    
    return dis_coordn


n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)

k=5
n_dimension=2

X_low=isomap(X,k,n_dimension)

# TODO: get low dimensional coordinates using sklearn
plt.scatter(X_low[:,0], X_low[:,1], c=color)
# TODO: Compare
isomap = Isomap(n_neighbors=k, n_components=n_dimension)
X_iso = isomap.fit_transform(X)
plt.scatter(X_iso[:,0],X_iso[:,1], c=color)




