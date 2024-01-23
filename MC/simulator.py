import numpy as np
from lattice import Lattice

tm_decay_rate = {0: 0.1, 1:0.1, 2: 0.1, 3: 0.1}
tm_order2_rate = {0: {1: 0.5}}

class Simulator():
    def __init__(self, lattice):
        self.lattice = lattice.deep_copy()
        self.t = 0
        pass

    def step(self, steps = 1):
        emission = []
        for _ in steps:
            emission_step = []
            np.random.shuffle(self.lattice.excited)
            for p in self.lattice.excited:
                if (p.type == 'Yb'):
                    if np.random.rand() < tm_decay_rate[p.state]:
                        emission_step.append() ## TODO
                        continue
                    p_space = 1-tm_decay_rate[p.state]
                    neighbors = self.lattice.neighbors[p]
                    ## TODO: how to calculate probabilities from rates?
                    for nei in neighbors:
                        if np.random.rand() < tm_order2_rate[p.state][nei.state] / p_space:
                            ## TODO: state transitions
                            break
                        else:
                            p_space *= (1-tm_order2_rate[p.state][nei.state])

                else:
                    pass
            self.t += 1

