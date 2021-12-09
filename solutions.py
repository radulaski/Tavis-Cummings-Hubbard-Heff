import numpy as np

#still needs to be verified for non PBC and loss terms
def identical_cavities(num_cavities, emitters_per_cavity, cavity_freq, emitter_freq, hopping, g, periodic = False, kappa=0, gamma=0):
    if periodic:
        k =  np.arange(1, num_cavities+1) * 2 * np.pi/num_cavities
    else:
        k = np.arange(1, num_cavities+1) * np.pi / (num_cavities + 1)
    left = 1/2 * (-2*hopping*np.cos(k)+(emitter_freq-1j/2*gamma)+(cavity_freq-1j/2*kappa))
    right = 1/2 * np.sqrt(abs((2*hopping*np.cos(k)+(emitter_freq-1j/2*gamma)-(cavity_freq-1j/2*kappa)))**2+4*emitters_per_cavity*g**2)
    y1 = left - right
    y2 = np.ones(num_cavities*(max(emitters_per_cavity-1, 0)))*(emitter_freq-1j/2*gamma)
    y3 = left + right
    return np.concatenate((y1,y2,y3))

