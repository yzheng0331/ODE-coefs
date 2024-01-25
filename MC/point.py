from utils import *

tag={'c0':10000,'c1':1000,'c2':1000,'c3':1000,'c4':1000,'k31':1000,'k41':1000,'k51':1000,
        'Ws':1000,
        'W10':1000,
        'W21':1000,'W20':1000,
        'W32':1000,'W31':1000,'W30':1000,
        'W43':1000,'W42':1000,'W41':1000,'W40':1000,
        'W54':1000,'W53':1000,'W52':1000,'W51':1000,'W50':1000,
        'W65':1000,'W64':1000,'W63':1000,'W62':1000,'W61':1000,'W60':1000,
        'W76':1000,'W75':1000,'W74':1000,'W73':1000,'W72':1000,'W71':1000,'W70':1000,
        'MPR21':1000,'MPR43':1000,
        'laser': 1000}


class Point():
    def __init__(self, coor, mol=None, state=None):
        self.p = coor
        self.type = mol
        self.state = state
    
    def __hash__(self):
        return hash(self.p)
    
    def change_state(self, new_state):
        self.state = new_state

    def to_euclidean(self):
        a, b, c = self.p
        return (0.596*a+0.5*0.596*b, 3**(1/2)/2*0.596*b, 0.353*c)

    def to(self, other):
        p1 = self.p
        p2 = other.p
        vec = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        evec = to_euclidean(vec)
        return e_distance(evec)
    
    def deep_copy(self):
        return Point(self.p, self.type, self.state)
    
    def react(self, other):
        # return rate, new states
        if self.type == 'Yb' and self.state == 1:
            if other.type == 'Yb':
                if other.state == 0:
                    return tag['c0'], (0, 1)
                return None
            else:
                if other.state == 0:
                    return tag['c1'], (0, 2)
                if other.state == 1:
                    return tag['c2'], (0, 4)
                if other.state == 3:
                    return tag['c3'], (0, 6)
                if other.state == 6:
                    return tag['c4'], (0, 7)
                else:
                    return None
                
        elif self.type == 'Tm' and other.type == 'Tm':
            if self.state == 3 and other.state == 0:
                return tag['k31'], (1, 1)
            if self.state == 6 and other.state == 0:
                return tag['k41'], (2, 3)
            if self.state == 7 and other.state == 0:
                return tag['k51'], (5, 3)
            
        return None
    
    def get_decay_rates(self):
        ret = []
        for i in range(self.state):
            ret.append(tag[f'W{self.state}{i}'])
        return ret

    def __str__(self):
        return f'{self.p} {self.type} {self.state}'
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.p[0] == other.p[0] and self.p[1] == other.p[1] and self.p[2] == other.p[2]
        return False
