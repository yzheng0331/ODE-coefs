import numpy as np
import random
from lattice import *
from point import *
from tqdm import tqdm


dt = 10**(-6)

class Simulator():
    def __init__(self, lattice):
        self.lattice = lattice.deep_copy()
        self.t = 0

    def step(self, steps = 1):
        for _ in range(steps):
            np.random.shuffle(self.lattice.excited)

            # excited state yb or tm state transition
            for p in self.lattice.excited:
                rates = []
                pair_rates = []
                neighbors = self.lattice.neighbors[p]
                for nei, _ in neighbors:
                    pair = p.react(nei)
                    if pair is not None:
                        rates.append(pair[0])
                        pair_rates.append((nei, pair[1]))

                p_decay_rates = p.get_decay_rates()
                no_reaction_prob = 1-dt*(sum(rates) + sum(p_decay_rates))

                # stay in current state
                if np.random.rand() < no_reaction_prob:
                    continue 

                # decay
                if np.random.rand() < sum(p_decay_rates) / (sum(rates) + sum(p_decay_rates)):
                    decayed = [i for i in range(p.state)]
                    decay_rates_sum = sum(p_decay_rates)
                    p_decay_rates = [i/decay_rates_sum for i in p_decay_rates]
                    new_state = np.random.choice(decayed, p=p_decay_rates)
                    p.state = new_state

                # etu
                else:
                    prob_sum = sum(rates)
                    rates = [i/prob_sum for i in rates]
                    # print(len(pair_rates), pair_rates)
                    # print(len(rates), rates)
                    nei, new_state = random.choices(pair_rates, rates)[0]
                    p.state = new_state[0]
                    nei.state = new_state[1]
                
            # laser excites ground state yb to excited yb
            for p in self.lattice.ground_yb:
                if np.random.rand() < dt*tag['laser']:
                    p.state = 1

            self.lattice.excited = [p for p in self.lattice.points if p.state != 0]
            self.lattice.ground_yb = [p for p in self.lattice.points if p.type == 'Yb'  and p.state == 0]
            self.t += 1
    
    def show_state(self):
        self.lattice.plot_3d_points_with_plotly()

    def simulate(self, t1, t2):
        for _ in tqdm(range(t1)):
            self.step()
        for _ in tqdm(range(t2-t1)):
            self.step()
            ## TODO: collect stats


# lattice = Lattice(0.5, 0.5, 2, 0.5)
# lattice.plot_3d_points_with_plotly()
# simulator = Simulator(lattice)
# simulator.show_state()
# simulator.step()
# simulator.show_state()
