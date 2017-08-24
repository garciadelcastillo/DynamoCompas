
import math

__author__     = ['Tomas Mendez Echenagucia <mtomas@ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'mtomas@ethz.ch'

def fitness(X,mod):
    n = len(X)

    totX = 0
    for i in range(1,n):
        totX  = totX + X[i]
    G = 1 + (9/(n-1))*totX

    H = 1- math.sqrt(X[0]/G)-((X[0]/G)*math.sin(10*math.pi*X[0]))
    
    fit = G*H
    
    return fit

