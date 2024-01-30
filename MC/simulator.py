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
        # TODO 进的能量，出的能量，和能量差
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

    def simulate(self, t1, t2=None):
        ## At 2500 steps, reach steady state
        ## TODO: 收集nir和blue的emission（其他能量也要记录、
        ## 折射率是1.5
        for _ in tqdm(range(t1)):
            self.step()
        if t2 is None:
            return
        c = 0
        yb_stats = []
        tm_stats = []
        for _ in tqdm(range(t2-t1)):
            self.step()
            c+=1
            if c%100 == 0:
                yb_stat, tm_stat = self.lattice.collect_stats()
                yb_stats.append(yb_stat)
                tm_stats.append(tm_stat)
        self.plot_stats(yb_stats, tm_stats)
    
    def plot_stats(self, yb_stats, tm_stats):
        print(yb_stats)

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
            plt.text(bar.get_x() + bar.get_width()/2, yval + 5, yb_avg[i], ha='center', va='bottom')

        # Plotting value distribution for type B using histogram
        # 1 row, 3 columns, 3rd plot
        tm_avg = []
        for i in range(len(tm_stats[0])):
            tm_avg.append(np.mean([j[i] for j in tm_stats]))
        # print(tm_avg)
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


# lattice = Lattice(0.5, 0.5, 2, 0.5)
# lattice.plot_3d_points_with_plotly()
# lattice.plot_3d_points_with_na()

# simulator = Simulator(lattice)
# simulator.show_state()
# simulator.simulate(3000,4000)
# simulator.show_state()
