import numpy as np
import random
from lattice import *
from point import *
from tqdm import tqdm

tag_default={'c0':7.025758333333333e-39, # Yb-Yb resonant energy transfer
             'c1':8.823270689202585e-42,'c2':2.824326729780273e-41,'c3':2.510909737310349e-42,'c4':2.5507997193509964e-43,
             'k31':2.0458454593341336e-41,'k41':7.299896312979955e-41,'k51':1.2897342736133983e-40,
        'Ws':1000,
        'W10':125.48935641249709,
        'W21':3.318715560788497 + 149977.8404029679,'W20':176.99746253145912 + 50.01921044404302,
        'W32':34.206376660350635 + 7.407650126658919,'W31':66.54090079350377,'W30':778.6223334161804,
        'W43':1000.49241968088766640 + 1768677.8208097615,'W42':146.53082969740504,'W41':258.72754779151234 + 58.98152867828142,'W40':1725.685918453449,
        'W54':0.013601242611778256 + 0.017876416530239997 + 156605871.04362732,'W53':5.142484506889417 + 230669.86963087242,'W52':192.81631278900016,'W51':362.10251583753916,'W50':548.8209590762159,
        'W65':12.27244074045102,'W64':10045.2434631327987160,'W63':23.045067137896037,'W62':494.8335554945873,'W61':790.6119449426245,'W60':612.1894036274351,
        'W76':95.08404006966971,'W75':686.9558866118873,'W74':488.5169554820362,'W73':2125.9773631820567,'W72':94.77917251652532,'W71':2862.4113298030165,'W70':7073.7489463917145,
        'MPR21':0,'MPR43':0,
        'laser': 1000}

dt = 10**(-6)

class Simulator():
    def __init__(self, lattice, tag = None):
        self.lattice = lattice.deep_copy()
        self.t = 0
        if tag is not None:
            self.tag = tag
        else:
            self.tag = tag_default

    def step(self, steps = 1, emission = False):
        # TODO 进的能量，出的能量，和能量差
        if emission:
            nirs = []
            blues = []
        for _ in range(steps):
            if emission:
                nir = 0
                blue = 0

            np.random.shuffle(self.lattice.excited)

            # excited state yb or tm state transition
            for p in self.lattice.excited:
                rates = []
                pair_rates = []
                neighbors = self.lattice.neighbors[p]
                for nei, distance in neighbors:
                    pair = p.react(nei, self.tag, distance)
                    if pair is not None:
                        rates.append(pair[0])
                        pair_rates.append((nei, pair[1]))

                p_decay_rates = p.get_decay_rates(self.tag)
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
                    if emission: ## TODO: 怎么从个数换算到intensity
                        if p.state == 3 and new_state == 0:
                            nir += 1
                        if p.state == 7 and new_state == 5:
                            nir += 1
                        if p.state == 6 and new_state == 2:
                            nir += 1
                        if p.state == 7 and new_state == 1:
                            blue += 1
                        if p.state == 6 and new_state == 0:
                            blue += 1
                    p.state = new_state

                # etu
                else:
                    prob_sum = sum(rates)
                    rates = [i/prob_sum for i in rates]
                    nei, new_state = random.choices(pair_rates, rates)[0]
                    p.state = new_state[0]
                    nei.state = new_state[1]
                
            # laser excites ground state yb to excited yb
            for p in self.lattice.ground_yb: 
                if np.random.rand() < dt*self.tag['laser']:
                    p.state = 1
            
            # update new excited state Yb and Tm, and update new ground state Yb
            self.lattice.excited = [p for p in self.lattice.points if p.state != 0]
            self.lattice.ground_yb = [p for p in self.lattice.points if p.type == 'Yb'  and p.state == 0]
            self.t += 1

            if emission:
                nirs.append(nir)
                blues.append(blue)
        if emission:
            if steps == 1:
                return nirs[0], blues[0]
            else: 
                return nirs, blues
    
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
        c = 0
        yb_stats = []
        tm_stats = []
        nirs = []
        blues = []
        for _ in tqdm(range(t2-t1)):
            nir, blue = self.step(emission = True)
            nirs.append(nir)
            blues.append(blue)
            c+=1
            if c%100 == 0:
                yb_stat, tm_stat = self.lattice.collect_stats()
                yb_stats.append(yb_stat)
                tm_stats.append(tm_stat)
        self.plot_stats(yb_stats, tm_stats)
        return np.mean(nirs), np.mean(blues)
    
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


# lattice = Lattice(0.5, 0.5, 2, 0.5)
# lattice.plot_3d_points_with_plotly()
# lattice.plot_3d_points_with_na()

# simulator = Simulator(lattice)
# simulator.show_state()
# simulator.simulate(3000,4000)
# simulator.show_state()
