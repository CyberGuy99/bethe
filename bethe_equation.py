from math import sqrt
import numpy as np
from scipy.optimize import root


class Bethe_Analytic:
    def m_1_solve(V, v):
        M = 1
        N = 2*M + 2*v
        num = lambda s: V*(1+2*v) + s*sqrt(V*V*(1+2*v)**2 + N**2)
        solns = [num(s)/(-1*N) for s in [1,-1]]
        return solns

    def m_2_solve(V,v):
        num = lambda s: V*(1+2*v) + s*sqrt(V*V*(1+2*v)**2 + 4*(1+v)**2)
        solns = [num(s)/(-2*(1+v)) for s in [1,-1]]
        return solns



def sgn(x):
    assert isinstance(x, (int,float)) 
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def _g(N, V, W):
    num = V*V - W*W
    denom = N*N * (sgn(num))
    if num == 0:
        raise ValueError
    return sqrt(num / denom)

def _eta(V, W):
    s = sgn(V*V - W*W)
    if s == 0 or V == W:
        raise ValueError
        
    return -1*sqrt( (V+W) / (s*(V-W)) )

def get_extra_params(N, V, W):
    return _g(N, V, W), _eta(V, W), sgn(V*V - W*W)


better_init = lambda M: [i/M for i in range(M)]

class Bethe_Equation_Solver:
    LARGE_INT = 9999
    POKE = 1e-4

    def __init__(self, N, V=1, W=0):
        self.set_params(N,V,W)

    def set_params(self, N, V, W):
        self.N = N
        self.inputs = Bethe_Equation_Solver.get_possible_inputs(N)
        self.V = V
        self.W = W
        self.g, self.eta, self.s = get_extra_params(N,V,W)
    
    def get_possible_inputs(N):
        # output list of inputs where 2*M + v_a + v_b = N
        assert type(N) is int
        if N % 2 == 1:
            M = N//2
            return [(M, 0, 1), (M, 1, 0)]

        return [(N//2, 0, 0), (N//2-1, 1, 1)]

    def solve_bethe(self, init_strat=better_init, method="hybr", verbose=False):
        E_options = []
        for M, v_a, v_b in self.inputs:
            solution = root(self.bethe_func, init_strat(M), args=(M, (v_a, v_b)), method=method)
            if verbose:
                print(solution)
            E = solution.x
            if verbose:
                print(E)
            E_options.append(E)

        return self.inputs, E_options

    def bethe_func(self, E, M, fiducial):
        assert len(E) == M, f'Incorrect input vector length (expected: {M}, actual: {len(E)})'
        v_a, v_b = fiducial
        
        full_out = [self.bethe_func_mini(E, l, M, v_a, v_b) for l in range(0, M)]
        return np.array(full_out)

    
    def bethe_func_mini(self, E, l, M, v_a, v_b):
        assert len(E) == M, f'Incorrect input vector length (expected: {M}, actual: {len(E)})'
        assert 2*M + v_a + v_b == self.N
            
        left = 1
        left_denom = (self.N * (E[l]*E[l] - self.eta*self.eta))
        if left_denom == 0:
            return LARGE_INT
            
        left_coeff = -1*self.eta/left_denom
        left += left_coeff*(self.g*self.N*(v_a - v_b)*(1 + self.s*E[l]*E[l]) + 2*self.V*E[l]*(1 + v_a + v_b))
        
        right = 0
        for n in range(0, M):
            if n == l:
                continue
            right_denom = (E[l] - E[n])
            if right_denom == 0:
                right_denom += POKE
                
            right += (1 + self.s*E[l]*E[n])/right_denom
    
        return left + (2*self.g*right)



TEST_THRESHOLD = 1e-5

def test_N_7():
    N = 7
    W = .5
    V = .75
    expected_roots = list(reversed([1.94591, 1.33363, .701066]))

    bethe_solver = Bethe_Equation_Solver(N=N, V=V, W=W)
    bethe_solver.inputs = bethe_solver.inputs[1:] # just run the (1,0) fiducial state
    params, root_lists = bethe_solver.solve_bethe(verbose=True)
    print('\nSpectral Parameters::')
    print(params[0], root_lists[0])
    
    diffs = [expected - actual for expected, actual in zip(expected_roots, root_lists[0])]
    if all(abs(d) < TEST_THRESHOLD for d in diffs):
        print('Good!')
    else:
        print('Inaccurate output!')
