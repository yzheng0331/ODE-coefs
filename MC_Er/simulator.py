import numpy as np
import random
from lattice import *
from point import *
from tqdm import tqdm
from squareLattice import SquareLattice

tag_default={'S1S0':100000, # energy transfer
     'c1':2.5*10**4,'c2':3.2*10**3, # upconversion
     'Ws':1000, # decay of sensitizer
     'W10':1000, 'W20':7000, # decay of activator
     'A1S0': 10000, # activator transfer energy back to sensitizer
     'A1A0': 5000, 'A1A1':600, # activator cross relaxation 
     'laser': 5.76*10**(-6)} # 100W
dt = 10**(-6)
# sqr = 0 # 10**5

class Simulator():
    def __init__(self, lattice, tag = None, sqr = 0):
        self.lattice = lattice.deep_copy()
        self.t = 0
        if tag is not None:
            self.tag = tag
        else:
            self.tag = tag_default
        self.sqr = sqr

    def step(self, steps = 1, ss = 0, emission = False):
        # TODO 进的能量，出的能量，和能量差
        ret = [0, 0, 0, 0, 0, 0] 
        Yb_0 = []
        Yb_1 = []
        Tm_0 = []
        Tm_1 = []
        Tm_2 = []
        # absorbed photons, recombined S1, recombined A1, recombined A2, uc
        for ii in tqdm(range(steps)):

            np.random.shuffle(self.lattice.excited)

            # excited state yb or tm state transition
            for p in self.lattice.excited:
                if p.state == 0:
                    continue
                rates = []
                pair_rates = []
                neighbors = self.lattice.neighbors[p]
                for nei in neighbors:
                    pair = p.react(nei, self.tag)
                    if pair is not None:
                        rates.append(pair[0])
                        pair_rates.append((nei, pair[1]))

                p_decay_rates = p.get_decay_rates(self.tag)
                if self.lattice.on_boundary(p):
                    p_decay_rates += self.sqr
                no_reaction_prob = 1-dt*(sum(rates) + p_decay_rates)

                # stay in current state
                if np.random.rand() < no_reaction_prob:
                    continue 

                # decay
                if np.random.rand() < p_decay_rates / (sum(rates) + p_decay_rates):
                    if p.type == 'Yb': 
                        if self.lattice.on_boundary(p) and np.random.rand() < self.sqr / p_decay_rates:
                            ret[5] += 1
                        else:
                            ret[1] += 1
                    else:
                        if self.lattice.on_boundary(p) and np.random.rand() < self.sqr / p_decay_rates:
                            ret[5] += 1
                        elif p.state == 1:
                            ret[2] += 1
                        else:
                            ret[3] += 1 
                    p.state = 0

                # etu
                else:
                    prob_sum = sum(rates)
                    rates = [i/prob_sum for i in rates]
                    nei, new_state = random.choices(pair_rates, rates)[0]
                    p.state = new_state[0]
                    nei.state = new_state[1]
                    if nei.state == 2 and p.type == 'Yb':
                        ret[4] += 1
                
            # laser excites ground state yb to excited yb
            for p in self.lattice.ground_yb: 
                if np.random.rand() < self.tag['laser']:
                    p.state = 1
                    ret[0] += 1
            
            # update new excited state Yb and Tm, and update new ground state Yb
            self.lattice.excited = [p for p in self.lattice.points if p.state != 0]
            self.lattice.ground_yb = [p for p in self.lattice.points if p.type == 'Yb'  and p.state == 0]
            self.t += 1 

            if ii >= ss:
                Yb_0.append(len([p for p in self.lattice.points if p.type == 'Yb' and p.state == 0]))
                Yb_1.append(len([p for p in self.lattice.points if p.type == 'Yb' and p.state == 1]))
                Tm_0.append(len([p for p in self.lattice.points if p.type == 'Tm' and p.state == 0]))
                Tm_1.append(len([p for p in self.lattice.points if p.type == 'Tm' and p.state == 1]))
                Tm_2.append(len([p for p in self.lattice.points if p.type == 'Tm' and p.state == 2]))
                # print(len([p for p in self.lattice.points if p.type == 'Yb' and p.state == 0]), 
                #       len([p for p in self.lattice.points if p.type == 'Yb' and p.state == 1]), 
                #       len([p for p in self.lattice.points if p.type == 'Tm' and p.state == 0]), 
                #       len([p for p in self.lattice.points if p.type == 'Tm' and p.state == 1]), 
                #       len([p for p in self.lattice.points if p.type == 'Tm' and p.state == 2]))
    
        return ret, Yb_0, Yb_1, Tm_0, Tm_1, Tm_2
    # [sum(Yb_0)/len(Yb_0), sum(Yb_1)/len(Yb_1),sum(Tm_0)/len(Tm_0), sum(Tm_1)/len(Tm_1), sum(Tm_2)/len(Tm_2)]
        
    
    def show_state(self):
        self.lattice.plot_3d_points_with_plotly()
    
    def plot_distributions(self):
        self.lattice.plot_distributions()

    def simulate(self, t1, t2=None):
        ## At 2500 steps, reach steady state
        ## 折射率是1.5
        for _ in tqdm(range(t1)):
            self.step()
        if t2 is None:
            return
        # c = 0
        # yb_stats = []
        # tm_stats = []
        # nirs = []
        # blues = []
        for _ in tqdm(range(t2-t1)):
            nir, blue = self.step(emission = True)
        #     nirs.append(nir)
        #     blues.append(blue)
        #     c+=1
        #     if c%100 == 0:
        #         yb_stat, tm_stat = self.lattice.collect_stats()
        #         yb_stats.append(yb_stat)
        #         tm_stats.append(tm_stat)
        # self.plot_stats(yb_stats, tm_stats)
        # return np.mean(nirs), np.mean(blues)
    
    def plot_stats(self, yb_stats, tm_stats):

        plt.figure(figsize=(15, 5))

        # 1 row, 3 columns, 1st plot
        plt.subplot(1, 3, 1)

        bars = plt.bar(['Yb', 'Tm', 'Y'], [self.lattice.yb_num, self.lattice.tm_num, self.lattice.n_points-self.lattice.yb_num-self.lattice.tm_num], color=['blue', 'pink', 'green'], width=0.4)
        plt.ylabel('Count',fontsize=18)
        plt.title('Distribution of three types',fontsize=18)
        plt.xticks(['Yb', 'Tm', 'Y'], ['Sensitizers', 'Emitters', 'Others'],fontsize=16)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom')

        # Plotting value distribution for type A using histogram
        # 1 row, 3 columns, 2nd plot
        yb_avg = []
        for i in range(len(yb_stats[0])):
            yb_avg.append(np.mean([j[i] for j in yb_stats]))
        plt.subplot(1, 3, 2)
        bars = plt.bar([0,1], yb_avg, color='blue', width=0.4)
        plt.ylabel('Count',fontsize=18)
        plt.title('Value distribution for sensitizers',fontsize=18)
        plt.xticks([0, 1], ['0(Ground state)', '1(Excited state)'],fontsize=16)
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yb_avg[i],1), ha='center', va='bottom')

        # Plotting value distribution for type B using histogram
        # 1 row, 3 columns, 3rd plot
        tm_avg = []
        for i in range(len(tm_stats[0])):
            tm_avg.append(np.mean([j[i] for j in tm_stats]))
        plt.subplot(1, 3, 3)
        bars = plt.bar([0,1,2,3,4,5,6,7], tm_avg, color='pink', width=0.4)
        plt.ylabel('Count',fontsize=18)
        plt.title('Value distribution for emitters',fontsize=18)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['G', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th'],fontsize=16)
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(tm_avg[i],1), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


# lattice = SquareLattice(0.2, 0.02)
# lattice.plot_3d_points_with_plotly()
# lattice.plot_distributions()
# lattice.plot_3d_points_with_na()

# simulator = Simulator(lattice)
# print(simulator.step(100, 0))
# simulator.show_state()
# simulator.simulate(3000,4000)
# simulator.show_state()
