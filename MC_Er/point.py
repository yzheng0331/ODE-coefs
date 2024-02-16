from utils import to_euclidean, e_distance


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
        return (a, b, c)

    def to(self, other):
        p1 = self.p
        p2 = other.p
        vec = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        evec = to_euclidean(vec)
        return e_distance(evec)
    
    def deep_copy(self):
        return Point(self.p, self.type, self.state)
    
    def react(self, other, tag):
        # return rate, new states
        if self.type == 'Yb' and self.state == 1:
            if other.type == 'Yb':
                if other.state == 0:
                    return tag['S1S0'], (0, 1)
                return None
            else:
                if other.state == 0:
                    return tag['c1'], (0, 1)
                if other.state == 1:
                    return tag['c2'], (0, 2)
                else:
                    return None
        elif self.type == 'Tm' and self.state == 1:
            if other.type == 'Yb':
                if other.state == 0:
                    return tag['A1S0'], (0, 1)
                return None
            else:
                if other.state == 0:
                    return tag['A1A0'], (0, 1)
                if other.state == 1:
                    return tag['A1A1'], (0, 2)
                return None
        return None
    
    def get_decay_rates(self, tag):
        return tag[f'W{self.state}0']

    def __str__(self):
        return f'{self.p} {self.type} {self.state}'
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.p[0] == other.p[0] and self.p[1] == other.p[1] and self.p[2] == other.p[2]
        return False
