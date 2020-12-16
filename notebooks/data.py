import numpy as np
from scipy.stats import norm

def synthesize_crosscurve(sample_nums=2000, z_list=None):
    u = lambda z: np.pi*(0.6+1.8*norm.cdf(z))
    f1 = lambda u: (1./np.sqrt(2))*(np.cos(u)/((np.sin(u))**2 + 1.))
    f2 = lambda u: (np.sqrt(2)) * (np.cos(u) * np.sin(u))/((np.sin(u))**2 + 1.)

    if z_list is None: z_list = np.random.normal(0,1,sample_nums)
    assert len(z_list) == sample_nums, "number of z_list should equal to sample_nums"

    u_list = u(z_list)
    f1_list = f1(u_list) + np.random.normal(0, np.sqrt(0.02), sample_nums)
    f2_list = f2(u_list) + np.random.normal(0, np.sqrt(0.02), sample_nums)
    return np.vstack([f1_list, f2_list]).T


def synthesize_threeclusters(sample_nums=2000, z_list=None):
    u = lambda z: (2*np.pi)/(1.+np.exp(-0.5*np.pi*z))
    t = lambda u: 2*np.tanh(10*u - 20*np.floor(u/2.) -10.) + 4.*np.floor(u/2.) + 2
    f1 = lambda t: np.cos(t)
    f2 = lambda t: np.sin(t)

    if z_list is None: z_list = np.random.normal(0,1,sample_nums)
    assert len(z_list) == sample_nums, "number of z_list should equal to sample_nums"

    u_list = u(z_list)
    t_list = t(u_list)
    f1_list = f1(t_list) + np.random.normal(0, np.sqrt(0.2), sample_nums)
    f2_list = f2(t_list) + np.random.normal(0, np.sqrt(0.2), sample_nums)
    return np.vstack([f1_list, f2_list]).T