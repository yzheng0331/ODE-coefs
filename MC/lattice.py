import numpy as np
import math
from point import Point
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Lattice():
    # shape of lattice based on https://www.nature.com/articles/s41598-018-19415-w and
    # https://www.researchgate.net/publication/253240215_Enhancement_of_blue_upconversion_luminescence_in_hexagonal_NaYF_4_YbTm_by_using_K_and_Sc_ions?enrichId=rgreq-ded66b2e7d92246868aa37d5e2ce7db2-XXX&enrichSource=Y292ZXJQYWdlOzI1MzI0MDIxNTtBUzoxMDIwMTIzNzI5MTQxNzlAMTQwMTMzMzA1MzY0Nw%3D%3D&el=1_x_3&_esc=publicationCoverPdf
    # Simplifying assumptions: in a unit cell, 
    #     xy projection: in the center of triangle formed by Na
    #     z projection: height is the middle point between 2 layers of Na

    # Some ignored details: probabilistic occupation of nodes, P\bar{6}2m, P\bar{6}, P6_3m
    #    https://www.researchgate.net/profile/Sameera-Perera-2/publication/318604006_Average_and_Local_Crystal_Structure_of_b-ErYbNaYF_4_Upconverting_Nanocrystals_Probed_by_X-ray_Total_Scattering/links/62f94e40b8dc8b4403e1987c/Average-and-Local-Crystal-Structure-of-b-ErYbNaYF-4-Upconverting-Nanocrystals-Probed-by-X-ray-Total-Scattering.pdf
    # TODO: do we need to modify the model?

    def __init__(self, yb_conc, tm_conc, d, r, seed = None):
        if seed is not None:
            np.random.seed(seed)

        # Create the lattice
        l_r = int(d/2/0.596)
        l_z = int(d/2/0.353)
        na_points = [Point((i, j, k), mol = 'Na') for i in range(-l_r, l_r+1) for j in range(-l_r, l_r+1) for k in range(-l_z, l_z+1)]
        y_coords = [Point((na.p[0]+1/4, na.p[1]+1/4, na.p[2]+1/4)) for na in na_points]
        na_points = self.in_diameter(d, na_points)
        y_coords = self.in_diameter(d, y_coords)
        n_points = len(y_coords)

        # Assign molecules to the lattice points
        n_yb = int(yb_conc*n_points)
        n_tm = int(tm_conc*n_points)
        types =  ['Yb'] * n_yb + ['Tm'] * n_tm + ['Y'] * (n_points - n_yb - n_tm)
        np.random.shuffle(types)
        values = []
        for p, t in zip(y_coords, types):
            p.type = t
            if t == 'Yb':
                p.state = np.random.choice([0, 1], p=[0.85, 0.15])
                # values.append(np.random.choice([0, 1], p=[0.85, 0.15]))  
                ### here, because of absorbation rate of Yb, set the rate manually as 0.85 and 0.15
            elif t == 'Tm':
                p.state = 0
                # values.append(0)  # All type B points have value 2
            else:
                p.state = 0
                # values.append(-1)  # No associated value for type C
        # y_points = [Point(coord, t, v) for t, coord, v in zip(types, y_coords, values)]
        y_points = y_coords
        
        self.yb_conc = yb_conc
        self.tm_conc = tm_conc 
        self.d = d
        self.r = r
        self.na_points = na_points
        self.points = y_points
        self.n_points = n_points
        self.get_neighbors(r)
        self.excited = [p for p in self.points if p.state != 0]
        self.ground_yb = [p for p in self.points if p.type == 'Yb'  and p.state == 0]
    
    def get_neighbors(self, r):
        # Get all neighbors (within distance r) of every point 
        ret = {p:[] for p in self.points}
        for i in range(self.n_points):
            i_nei = []
            for j in range(self.n_points):
                if i == j :
                    continue
                dist = self.points[i].to(self.points[j])
                if dist <= r:
                    i_nei.append(self.points[j])
            ret[self.points[i]] = (i_nei, dist)
        self.neighbors = ret

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
        counts, bins, patches = plt.hist(all_values_B, bins=[0, 1, 2, 3, 4, 5], align='left', rwidth=0.4, color='pink')

        plt.ylabel('Count',fontsize=18)
        plt.title('Value distribution for emitters',fontsize=18)
        plt.xticks([0, 1, 2, 3, 4], ['G', '1st', '2nd', '3rd', '4th'],fontsize=16)
        for count, bin, patch in zip(counts, bins, patches):
            plt.text(bin + 0.01, count + 1, int(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

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
                            text=values_A, textposition='top center')

        trace_B = go.Scatter3d(x=x_B, y=y_B, z=z_B, mode='markers+text',
                            marker=dict(size=6, color='pink', opacity=0.8),
                            text=values_B, textposition='top center')

        # Combine plots and set layout
        data = [trace_A, trace_B]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)

        # Display the figure
        fig.show()

    def plot_3d_points_with_na(self):
        euclidean_coords_na = [(point.to_euclidean()) for point in self.na_points]
        na_x = [point[0] for point in euclidean_coords_na]
        na_y = [point[1] for point in euclidean_coords_na]
        na_z = [point[2] for point in euclidean_coords_na]
        trace_na = go.Scatter3d(x=na_x, y=na_y, z=na_z, mode='markers+text',
                            marker=dict(size=3, color='black', opacity=0.8),
                            text='Na', textposition='top center')

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
                            text=values_A, textposition='top center')

        trace_B = go.Scatter3d(x=x_B, y=y_B, z=z_B, mode='markers+text',
                            marker=dict(size=6, color='pink', opacity=0.8),
                            text=values_B, textposition='top center')

        # Combine plots and set layout
        data = [trace_A, trace_B, trace_na]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)

        # Display the figure
        fig.show()

    def deep_copy(self):
        # Create deep copy of lattice so that we can perform experiments with the same initial state
        # ALERT: na_points is not deep copied
        cp = Lattice(self.yb_conc, self.tm_conc, self.d, seed=np.random.get_state())
        cp.yb_conc = self.yb_conc
        cp.tm_conc = self.tm_conc 
        cp.d = self.d
        cp.r = self.r

        cp.na_points = self.na_points
        cp.points = [p.deep_copy() for p in self.y_points]
        cp.n_points = self.n_points
        cp.get_neighbors(cp.r)
        cp.excited = [p for p in cp.points if p.state != 0]
        cp.ground_yb = [p for p in cp.points if p.type == 'Yb'  and p.state == 0]


lattice = Lattice(0.5, 0.5, 2, 1)
lattice.plot_distributions()
lattice.plot_3d_points_with_na()