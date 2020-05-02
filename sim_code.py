# Initial code from https://rajeshrinet.github.io/blog/2014/ising-model/

import numpy as np 
from numpy.random import rand
from scipy.special import binom
from unionfind import UnionFind

# Implement Union Find to find clusters in the configurations
# https://github.com/deehzee/unionfind/blob/master/unionfind.py

def initial_config(N, no_colors=2):   
    ''' Generate a random color/spin configuration for initialization'''
    state = np.random.randint(low=1, high=no_colors+1, size=(N,N))
    return state


def mc_move(config, eta_prob, eta, N, no_colors, sites):
    '''Monte Carlo move using generalized SW algorithm '''
    # Assign eta to each hyperedge
    eta = assign_etas(config, eta_prob, eta, no_colors)
    # print('assigned eta:', eta)

    # Assign labels to each site
    config = assign_labels(config, eta, N, no_colors, sites)
    # print('assigned configuration:\n', config)

    return config


def number_of_colors (config):
    '''Number of colors used in the given confuration of labels'''
    return len(np.unique(config))


def avg_sites_per_color (config):
    '''Average number of sites each present color has in the configuration'''
    unq = np.unique(config)
    return config.shape[0]*config.shape[1]/len(unq)


def prob_eta_lambda(gamma, no_colors, k):
    '''Probability of the eta lambda corresponding to a certain number of colors'''
    if k < no_colors:
        return ( np.exp((no_colors-k)*gamma) - np.exp((no_colors-k-1)*gamma) ) / np.exp((no_colors-1)*gamma)
    elif k == no_colors:
        return 1 / np.exp((no_colors-1)*gamma)
    
    
def prob_eta_edge(J, col1, col2):
    '''Probability of the eta edge corresponding to open or not'''
    if col1==col2:
        return 1 - np.exp(-2*J)
    else:
        return np.exp(-2*J)
    
    
def prob_eta_site(alpha, no_colors, k):
    '''Probability of the eta site corresponding to a maximum color number (lower have higher energy)'''
    if k < no_colors:
        return ( np.exp((no_colors-k)*alpha) - np.exp((no_colors-k-1)*alpha) ) / np.exp((no_colors-1)*alpha)
    elif k == no_colors:
        return 1 / np.exp((no_colors-1)*alpha)

    
def assign_etas(config, eta_prob, eta, no_colors):
    
    '''Assign a number of colors that is at least the current number of colors'''
    eta_lambda = eta[0]
    prob_lambda = eta_prob[0]
    # print('probabilities for no. of colors:', p)
    
    current_k = number_of_colors(config)
    p = prob_lambda[current_k-1:]
    p = p / p.sum()
    # print('normalized:', p)
    eta_lambda = np.random.choice((np.arange(current_k, no_colors+1)), p=p)
    eta[0] = eta_lambda

    '''Assign a closed or open bond to each edge'''
    eta_edges = eta[1]
    prob_edge = eta_prob[1]
    # print('probabilities for edges:', p)
    
    # loop over edges
    for i in range(eta_edges.shape[0]):
        for j in range(eta_edges.shape[1]):
            for e in range(eta_edges.shape[2]):
                if eta_edges[i,j,e] == -1: continue
                if e==0: current_b = (0 if config[i,j]==config[i+1,j] else 1)          
                elif e==1: current_b = (0 if config[i,j]==config[i,j+1] else 1) 
                    
                p = prob_edge[current_b:]
                p = p / p.sum()
                # print('normalized:', p)
                eta_edge = np.random.choice((np.arange(current_b, 1+1)), p=p) # 0 for present bonds, 1 for not
                eta_edges[i,j,e] = eta_edge
    eta[1] = eta_edges
    
    '''Assign a color that has at most the current energy for the color (color number is at least the current one)'''
    eta_sites = eta[2]
    prob_site = eta_prob[2]
    
    # loop over sites
    for i in range(eta_sites.shape[0]):
        for j in range(eta_sites.shape[1]):
            current_s = config[i,j]
            p = prob_site[current_s-1:]
            p = p / p.sum()
            # print('normalized:', p)
            eta_site = np.random.choice((np.arange(current_s, no_colors+1)), p=p)
            eta_sites[i,j] = eta_site
    eta[2] = eta_sites
            
    return eta


def site2str(tup):
    return str(tup[0]) + ',' + str(tup[1])

def str2site(strr):
    t = strr.split(',')
    return [int(x) for x in t]

def sample_config(config, eta, N, no_colors, sites):
        
    '''Generate clusters from the assigned bonds (eta_edge)'''
    uf = UnionFind(sites)
    eta_edges = eta[1]
    for i in range(eta_edges.shape[0]):
        for j in range(eta_edges.shape[1]):
            for e in range(eta_edges.shape[2]):
                if eta_edges[i,j,e] == -1: continue
                if eta_edges[i,j,e] == 0:
                    if e==0: uf.union(site2str((i,j)), site2str((i+1,j)))
                    elif e==1: uf.union(site2str((i,j)), site2str((i,j+1)))    

    '''For each cluster, find the site with strongest constraint (smallest eta_site)
       and assign that eta_site to the entire cluster
    '''
    eta_sites = eta[2]
    comp_constraints = {}
    for comp in uf.components():
        min_constraint = no_colors
        comp_root = '-1,-1'
        for site_str in comp:
            site = str2site(site_str)
            if eta_sites[site[0],site[1]] <= min_constraint:
                min_constraint = eta_sites[site[0],site[1]]
                comp_root = site_str
        comp_constraints[comp_root] = min_constraint       
    
    '''Randomly sample a color for each cluster'''
    for root in comp_constraints:
        max_col = comp_constraints[root]
        comp_color = np.random.choice((np.arange(1, max_col+1)))
        for site_str in uf.component(root):
            site = str2site(site_str)
            config[site[0],site[1]] = comp_color
            
    return config

    
def assign_labels(config, eta, N, no_colors, sites):
    '''Assign a color configuration chosen uniformly from the configurations compatible with eta'''
    # Brute force version
    max_colors = eta[0] # eta_lambda
    config = sample_config(config, eta, N, no_colors, sites)
    while number_of_colors(config) > max_colors:
        config = sample_config(config, eta, N, no_colors, sites)
    return config


Small experiment

N, q = 15, 12
gamma = 0.2
J = 0.5
alpha = 0.2

'''Probabilities ordered from highest (high energy) to lowest'''
lambda_prob = np.zeros(q)
for j in range(q):
    lambda_prob[j] = prob_eta_lambda(gamma,q,j+1)
print('lambda probabilities:', lambda_prob)

edge_prob = np.zeros(2)
edge_prob[0] = prob_eta_edge(J, 'same_col', 'same_col')
edge_prob[1] = prob_eta_edge(J, 'same_col', 'dif_col')
print('edge probabilities:', edge_prob)

site_prob = np.zeros(q)
for j in range(q):
    site_prob[j] = prob_eta_site(alpha,q,j+1)
print('site probabilities:', site_prob)

eta_prob = (lambda_prob, edge_prob, site_prob)

'''Current states of eta'''
eta_lambda = 0
eta_edges = np.zeros((N,N,2))
# Special edge cases (no neighbors at the border)
eta_edges[:,N-1,1] = -1
eta_edges[N-1,:,0] = -1
eta_sites = np.zeros((N,N))
eta = [eta_lambda, eta_edges, eta_sites]


'''List of sites (tuples) for Union-Find'''
sites = []
for i in range(N):
    for j in range(N):
        sites.append(str(i)+','+str(j))

config = initial_config(N,q)
print('\ninitial config:')
print(config)
config = mc_move(config, eta_prob, eta, N, q, sites)
print('\nnext config:')
print(config)

for i in range(1000):
    config = mc_move(config, eta_prob, eta, N, q, sites)
    if i%100 == 0: print(i)

print('\nfinal config:')
print(config)
