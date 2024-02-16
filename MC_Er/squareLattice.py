import numpy as np
import math
from point import *
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from scipy.integrate import odeint

class SquareLattice():
    
    def __init__(self,yb_conc,tm_conc):

        ## Assume 20% and 2%
        coordToPoints = {(i,j,k): Point((i,j,k)) for i in range(25) for j in range(25) for k in range(25)}
        points = list(coordToPoints.values())
        n_points = len(points)
        n_yb = int(n_points*10/11)
        n_tm = n_points - n_yb

        # Assign molecules to the lattice points 
        types = ['Yb'] * n_yb + ['Tm'] * n_tm 
        np.random.shuffle(types)
        for p, t in zip(points, types):
            p.type = t
            if t == 'Yb':
                p.state = np.random.choice([0, 1], p=[0.994, 0.006])
            else:
                p.state = 0
        self.coordToPoints = coordToPoints        
        self.yb_conc = yb_conc
        self.tm_conc = tm_conc 
        self.yb_num = n_yb
        self.tm_num = n_tm
        self.points = points # rare earth doping points, Yb/Tm
        self.n_points = len(self.points) # number of Yb/Tm points
        self.get_neighbors()
        self.excited = [p for p in self.points if p.state != 0]
        self.ground_yb = [p for p in self.points if p.type == 'Yb'  and p.state == 0]
    
    def get_neighbors(self,):
        # Get all neighbors (nearest neighbors) of every point 
        ret = {p:[] for p in self.points}
        nei_vecs = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
        for i in tqdm(range(self.n_points)):
            i_nei = []
            for j in nei_vecs:
                nei_coord = (self.points[i].p[0]+j[0], self.points[i].p[1]+j[1], self.points[i].p[2]+j[2])
                if nei_coord[0] <0 or nei_coord[0] >= 25:
                    continue
                if nei_coord[1] <0 or nei_coord[1] >= 25:
                    continue
                if nei_coord[2] <0 or nei_coord[2] >= 25:
                    continue
                i_nei.append(self.coordToPoints[nei_coord])
            ret[self.points[i]] = i_nei
        self.neighbors = ret
    

    
    # def valid_point(self, point):
    #     if point.p.x>=25 or point.p.x<0:
    #         return False
    #     if point.p.y>=25 or point.p.y<0:
    #         return False
    #     if point.p.z>=25 or point.p.z<0:
    #         return False
    #     return True

    def in_diameter(self, d, points):
        origin = Point((0,0,0))
        ret = []
        for point in points:
            if point.to(origin) < d/2:
                ret.append(point)
        return ret

    def plot_distributions(self):
        points = self.points
        # once we input the configurations, then we can get the total information of the system

        # Create a list of all types and values for easier plotting
        all_types = [point.type for point in points]
        all_values_A = [point.state for point in points if point.type == 'Yb']
        all_values_B = [point.state for point in points if point.type == 'Tm']

        plt.figure(figsize=(15, 5))

        # Plotting distribution of A, B, C using bar plot

        # 1 row, 3 columns, 1st plot
        plt.subplot(1, 3, 1)
        labels, counts = np.unique(all_types, return_counts=True)
        counts_tmp = dict(zip(labels, counts))
        counts_tmp['Y'] = counts_tmp.setdefault('Y', 0)

        bars = plt.bar(['Yb', 'Tm', 'Y'], [counts_tmp['Yb'], counts_tmp['Tm'], counts_tmp['Y']], color=['blue', 'pink', 'green'], width=0.4)
        plt.ylabel('Count',fontsize=18)
        plt.title('Distribution of three types',fontsize=18)
        plt.xticks(['Yb', 'Tm', 'Y'], ['Sensitizers', 'Emitters', 'Others'],fontsize=16)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom')

        # Plotting value distribution for type A using histogram
        # 1 row, 3 columns, 2nd plot
        plt.subplot(1, 3, 2)
        counts, bins, patches = plt.hist(all_values_A, bins=[0, 1, 2], align='left', rwidth=0.4, color='blue')
        plt.ylabel('Count',fontsize=18)
        plt.title('Value distribution for sensitizers',fontsize=18)
        plt.xticks([0, 1], ['0(Ground state)', '1(Excited state)'],fontsize=16)
        for count, bin, patch in zip(counts, bins, patches):
            plt.text(bin + 0.01, count + 1, int(count), ha='center', va='bottom')

        # Plotting value distribution for type B using histogram
        # 1 row, 3 columns, 3rd plot
        plt.subplot(1, 3, 3)
        counts, bins, patches = plt.hist(all_values_B, bins=[0, 1, 2], align='left', rwidth=0.4, color='pink')

        plt.ylabel('Count',fontsize=18)
        plt.title('Value distribution for emitters',fontsize=18)
        plt.xticks([0, 1, 2], ['G', '1st', '2nd'],fontsize=16)
        for count, bin, patch in zip(counts, bins, patches):
            plt.text(bin + 0.01, count + 1, int(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
    
    def collect_stats(self):
        yb_1 = len([i for i in self.points if i.type == 'Yb' and i.state == 1])
        yb_0 = len([i for i in self.points if i.type == 'Yb' and i.state == 0])

        tm_0 = len([i for i in self.points if i.type == 'Tm' and i.state == 0])
        tm_1 = len([i for i in self.points if i.type == 'Tm' and i.state == 1])
        tm_2 = len([i for i in self.points if i.type == 'Tm' and i.state == 2])
        tm_3 = len([i for i in self.points if i.type == 'Tm' and i.state == 3])
        tm_4 = len([i for i in self.points if i.type == 'Tm' and i.state == 4])
        tm_5 = len([i for i in self.points if i.type == 'Tm' and i.state == 5])
        tm_6 = len([i for i in self.points if i.type == 'Tm' and i.state == 6])
        tm_7 = len([i for i in self.points if i.type == 'Tm' and i.state == 7])

        return [yb_0, yb_1], [tm_0, tm_1, tm_2, tm_3, tm_4, tm_5, tm_6, tm_7]

    def plot_3d_points_with_plotly(self):
        points = self.points
        # Separate points based on their type (A or B)
        points_A = [point for point in points if point.type == 'Yb']
        points_B = [point for point in points if point.type == 'Tm']

        # Extract coordinates and values for points of type A
        euclidean_coords_A = [(point.to_euclidean()) for point in points_A]
        x_A = [point[0] for point in euclidean_coords_A]
        y_A = [point[1] for point in euclidean_coords_A]
        z_A = [point[2] for point in euclidean_coords_A]
        values_A = [point.state for point in points_A]

        # Extract coordinates and values for points of type B
        euclidean_coords_B = [(point.to_euclidean()) for point in points_B]
        x_B = [point[0] for point in euclidean_coords_B]
        y_B = [point[1] for point in euclidean_coords_B]
        z_B = [point[2] for point in euclidean_coords_B]
        values_B = [point.state for point in points_B]

        # Create 3D scatter plots
        trace_A = go.Scatter3d(x=x_A, y=y_A, z=z_A, mode='markers+text',
                            marker=dict(size=6, color='blue', opacity=0.8),
                            text=values_A, textposition='top center',
                            name = 'Yb')

        trace_B = go.Scatter3d(x=x_B, y=y_B, z=z_B, mode='markers+text',
                            marker=dict(size=6, color='pink', opacity=0.8),
                            text=values_B, textposition='top center',
                            name = 'Tm')

        # Combine plots and set layout
        data = [trace_A, trace_B]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(legend=dict(title='Legend'))
        fig.layout.scene.camera.projection.type = "orthographic"

        # Display the figure
        fig.show()
        fig.write_html("small.html")
    
    def plot_3d_points_with_na(self):
        euclidean_coords_na = [(point.to_euclidean()) for point in self.na_points]
        na_x = [point[0] for point in euclidean_coords_na]
        na_y = [point[1] for point in euclidean_coords_na]
        na_z = [point[2] for point in euclidean_coords_na]
        trace_na = go.Scatter3d(x=na_x, y=na_y, z=na_z, mode='markers+text',
                            marker=dict(size=4, color='green', opacity=0.8),
                            text='Na', textposition='top center', name = 'Na')

        points = self.points
        # Separate points based on their type (A or B)
        points_A = [point for point in points if point.type == 'Yb']
        points_B = [point for point in points if point.type == 'Tm']
        points_Y = [point for point in self.y_points if point.type == 'Y']

        # Extract coordinates and values for points of type A
        euclidean_coords_A = [(point.to_euclidean()) for point in points_A]
        x_A = [point[0] for point in euclidean_coords_A]
        y_A = [point[1] for point in euclidean_coords_A]
        z_A = [point[2] for point in euclidean_coords_A]
        values_A = [point.state for point in points_A]

        # Extract coordinates and values for points of type B
        euclidean_coords_B = [(point.to_euclidean()) for point in points_B]
        x_B = [point[0] for point in euclidean_coords_B]
        y_B = [point[1] for point in euclidean_coords_B]
        z_B = [point[2] for point in euclidean_coords_B]
        values_B = [point.state for point in points_B]

        # Extract coordinates and values for points of type Y
        euclidean_coords_Y = [(point.to_euclidean()) for point in points_Y]
        x_Y = [point[0] for point in euclidean_coords_Y]
        y_Y = [point[1] for point in euclidean_coords_Y]
        z_Y = [point[2] for point in euclidean_coords_Y]

        trace_Y = go.Scatter3d(x=x_Y, y=y_Y, z=z_Y, mode='markers',
                            marker=dict(size=6, color='gray', opacity=0.8),
                            textposition='top center', name = 'Y')

        # Create 3D scatter plots
        trace_A = go.Scatter3d(x=x_A, y=y_A, z=z_A, mode='markers+text',
                            marker=dict(size=6, color='blue', opacity=0.8),
                            text=values_A, textposition='top center',
                            name = 'Yb')

        trace_B = go.Scatter3d(x=x_B, y=y_B, z=z_B, mode='markers+text',
                            marker=dict(size=6, color='pink', opacity=0.8),
                            text=values_B, textposition='top center',
                            name = 'Tm')

        # Combine plots and set layout
        data = [trace_A, trace_B, trace_Y, trace_na]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.update_layout(legend=dict(title='Legend'))

        # Display the figure
        fig.show()

    def deep_copy(self):
        # Create deep copy of lattice so that we can perform experiments with the same initial state
        # ALERT: na_points is not deep copied

        # print(np.random.get_state())
        cp = SquareLattice(self.yb_conc, self.tm_conc)
        cp.yb_conc = self.yb_conc
        cp.tm_conc = self.tm_conc 

        cp.points = [p.deep_copy() for p in self.points]
        cp.coordToPoints = {}
        for p in cp.points:
            cp.coordToPoints[p.p] = p
        cp.n_points = self.n_points
        cp.get_neighbors()
        cp.excited = [p for p in cp.points if p.state != 0]
        cp.ground_yb = [p for p in cp.points if p.type == 'Yb'  and p.state == 0]

        return cp
    
    def on_boundary(self, point):
        if point.p[0] == 0 or point.p[0] == 24:
            return True
        if point.p[1] == 0 or point.p[1] == 24:
            return True
        if point.p[2] == 0 or point.p[2] == 24:
            return True
        return False

    
    def ode_distribution(self):
        ## Why ODE and MC doesn't match? presence of c0
        n_yb = int(self.yb_conc*self.n_points)
        n_tm = int(self.tm_conc*self.n_points)

        def system(state, t, tag):

            ns2, n0, n1, n2, n3, n4, n5, n6, n7 = state
            ms2 = tag['laser']*(n_yb-ns2) - tag['Ws']*ns2 - (tag['c1']*n0+tag['c2']*n1+tag['c3']*n3+tag['c1']*n6)*ns2
            m0 = -tag['c1']*n0*ns2                   + tag['W10']*n1 + tag['W20']*n2 + tag['W30']*n3 + tag['W40']*n4 + tag['W50']*n5 + tag['W60']*n6 + tag['W70']*n7 - tag['k31']*n0*n3 - tag['k41']*n0*n6 - tag['k51']*n0*n7
            m1 = -tag['c2']*n1*ns2 + tag['MPR21']*n2 - tag['W10']*n1 + tag['W21']*n2 + tag['W31']*n3 + tag['W41']*n4 + tag['W51']*n5 + tag['W61']*n6 + tag['W71']*n7 + tag['k31']*n0*n3
            m2 = tag['c1']*n0*ns2 - tag['MPR21']*n2  - tag['W20']*n2 - tag['W21']*n2 + tag['W32']*n3 + tag['W42']*n4 + tag['W52']*n5 + tag['W62']*n6 + tag['W72']*n7 + tag['k31']*n0*n3 + tag['k41']*n0*n6
            m3 = tag['c3']*n3*ns2 + tag['MPR43']*n4  - tag['W30']*n3 - tag['W31']*n3 - tag['W32']*n3 + tag['W40']*n4 + tag['W53']*n5 + tag['W63']*n6 + tag['W73']*n7 - tag['k31']*n0*n3 + tag['k41']*n0*n6 + tag['k51']*n0*n7
            m4 = tag['c2']*n1*ns2 - tag['MPR43']*n4  - tag['W40']*n4 - tag['W41']*n4 - tag['W42']*n4 - tag['W43']*n4 + tag['W54']*n5 + tag['W64']*n6 + tag['W74']*n7 
            m5 =                                     - tag['W50']*n5 - tag['W51']*n5 - tag['W52']*n5 - tag['W53']*n5 - tag['W54']*n5 + tag['W65']*n6 + tag['W75']*n7 + tag['k51']*n0*n7
            m6 = tag['c3']*n3*ns2 - tag['c4']*n6*ns2 - tag['W60']*n6 - tag['W61']*n6 - tag['W62']*n6 - tag['W63']*n6 - tag['W64']*n6 - tag['W65']*n6 + tag['W76']*n7 - tag['k41']*n0*n6
            m7 = tag['c4']*n6*ns2                    - tag['W70']*n7 - tag['W71']*n7 - tag['W72']*n7 - tag['W73']*n7 - tag['W74']*n7 - tag['W75']*n7 - tag['W76']*n7 - tag['k51']*n0*n7

            return [ms2, m0, m1, m2, m3, m4, m5, m6, m7]
        
        yb_excited = len([i for i in self.y_points if i.type == 'Yb' and i.state == 1])
        state0 = [yb_excited, n_tm, 0, 0, 0, 0, 0, 0, 0]
        t = np.arange(0.0, 0.001, 0.000001)
        state = odeint(system, state0, t)

        state_f = [state[:, 0][-1], state[:, 1][-1], state[:, 2][-1], state[:, 3][-1], state[:, 4][-1], state[:, 5][-1], state[:, 6][-1], state[:, 7][-1], state[:, 8][-1]]
        
        plt.figure(figsize=(5, 5))
        bars = plt.bar([0, 1, 2, 3, 4, 5, 6, 7], state_f[1:], width=0.4, color='pink')
        plt.ylabel('Count',fontsize=18)
        plt.title('ODE value distribution for emitters',fontsize=18)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['G', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th'],fontsize=16)
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(state_f[i+1],1), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        return state_f



# lattice = SquareLattice(0.2, 0.02)
# p0 = lattice.coordToPoints[(0,0,0)]
# p1 = lattice.neighbors[p0][0]
# print(len([p for p in lattice.points if p.state != 0]))
# print(p1.state)
# p1.state = 1
# print(len([p for p in lattice.points if p.state != 0]))

# print(lattice.ode_distribution())
# lattice.plot_distributions()
# lattice.plot_3d_points_with_plotly()
# lattice.plot_3d_points_with_na()

