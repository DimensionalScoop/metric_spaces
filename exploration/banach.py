import numpy as np
import matplotlib.pyplot as plt
import joblib
from joblib import delayed, Parallel

executer = Parallel(n_jobs=20)

DIMS = 20
N_PIVOTS = 20
pivots = np.random.rand(N_PIVOTS, DIMS) * 37*2 - 37
# pivots = np.diag([40]*DIMS)
v = np.random.rand(DIMS, 100_000) * 37*2 - 37
u = np.random.rand(DIMS, 100_000) * 37*2 - 37

def d(x,y):
    p = 2 
    summands = (np.abs(x-y))**p
    return np.sum(summands, axis=0)**(1/p)

def D(x,y):
    component_diffs = np.abs(x-y)
    return np.max(component_diffs, axis=0)

def D_min(x,y):
    component_diffs = np.abs(x-y)
    return np.min(component_diffs, axis=0)

def phi(x):
    jobs = [delayed(d)(p.reshape(-1,1), x) for p in pivots]
    res = executer(jobs)
    return np.vstack(list(res))

pp_dist = [[d(p1,p2) for p1 in pivots] for p2 in pivots]
pp_dist = np.asarray(pp_dist)


d(v, v[0]).shape
phi(v[:,:]).shape

# contraction factor
k = D(phi(v), phi(u)) / d(u,v)#  / d(v,u) 
k.mean()
k.std()
k.max()
k.min()

plt.hist(d(u,v), bins=100)
plt.hist(D(phi(u),phi(v)), bins=100, histtype='step')
plt.grid()
plt.savefig("/tmp/hist.svg")
