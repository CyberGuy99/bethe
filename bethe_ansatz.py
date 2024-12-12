from sys import argv
from math import acos, pi, sqrt, prod
from bethe_equation import Bethe_Equation_Solver, _eta, sgn

def _q(M, j, C):
    num = C[M + 2 - j - 1] # 1 based indexing
    denom = sqrt(sum([c*c for c in C[:M + 2 - j]]))
    return num/denom

def _w(M, v_a, j, i):
    return 2*M + v_a - 2*j + i

def _x(v_b, j, i):
    return v_b + 2*j + i

    
class Bethe_Ansatz:
    coeff_options = []
    angle_options = []

    def __init__(self, N, V, W, params_and_E_options=None, verbose=False):
        self.N = N
        self.V = V
        self.W = W
        self.verbose = verbose
        
        eta = _eta(V, W)
        if params_and_E_options:
            params, E_options = params_and_E_options
        else:
            bethe_solver = Bethe_Equation_Solver(N=N, V=V, W=W)
            params, E_options = bethe_solver.solve_bethe()
            
        for param, E_option in zip(params, E_options):
            M, v_a, v_b = param
            C = Bethe_Ansatz._compute_ansatz_coeff(M, E_option, v_a, v_b, eta)
            self.coeff_options.append(C)
    
            curr_angles = Bethe_Ansatz._compute_all_theta(M, C)
            self.angle_options.append(curr_angles)
            if self.verbose:
                print(curr_angles)

    def get_state(self, index=0):
        if index < len(self.coeff_options):
            return self.coeff_options[index]
        return []

    def get_prep_angles(self, index=0):
        if index < len(self.angle_options):
            return self.angle_options[index]
        return []
    
    
    # M represents M+1 in the inductive formula
    def _compute_intermediate_coeff(M, E, v_a, v_b, d_prev, eta=-1):
        amps = dict()
        state_order = []
        for j in range(M):
            d_j = d_prev[j]
            inc_a_state = (_w(M-1, v_a, j, 2), _x(v_b, j, 0))
            # E index should match the current EGO iter (M), but also have 1-indexing
            inc_a_coeff = prod(sqrt(_w(M-1, v_a, j, i+1)) for i in range(2)) / (E[M-1] + eta)
            
            inc_b_state = (_w(M-1, v_a, j, 0), _x(v_b, j, 2))
            inc_b_coeff = prod(sqrt(_x(v_b, j, i+1)) for i in range(2)) / (E[M-1] - eta)
    
            for state, coeff in [(inc_a_state, inc_a_coeff), (inc_b_state, inc_b_coeff)]:
                if state not in amps:
                    state_order.append(state)
                    amps[state] = 0
                amps[state] += d_j*coeff
            # print(amps)
            
        
        # assert output length is M+1
        assert len(state_order) == M+1, f'Unexpected state length: {len(state_order)}'
    
        mag_squared = sum(a*a for a in amps.values())
        mag_squared = 1
        print(state_order)
        return [amps[state]/sqrt(mag_squared) for state in state_order]
            
    def _compute_ansatz_coeff(M, E, v_a, v_b, eta=-1):
        D = []
        for l in range(1, M+1):
            # in lth iteration, prev output vector is a superposition of l fiducial states
            d_prev = D[-1] if len(D) > 0 else [1]
            # print(f'Norm squared after iter {l-1}: {sum(d_j * d_j for d_j in d_prev)}')
            D.append(Bethe_Ansatz._compute_intermediate_coeff(l, E, v_a, v_b, d_prev, eta=eta))
    
        m_sq = sum(d*d for d in D[-1])
        return [d / sqrt(m_sq) for d in D[-1]]
    
    def _compute_theta(M, j, C):
        if j > M:
            raise ValueError
        if j < M:
            return 2*acos(_q(M, j, C))
        return 2*sgn(C[0])*(acos(_q(M, j, C)) - pi) + 2*pi
    
    def _compute_all_theta(M, C):
        return [Bethe_Ansatz._compute_theta(M, j, C) for j in range(1, M+1)]
    


def main(N, verbose):
    params, root_lists = Bethe_Equations.solve_bethe(N=N, verbose=verbose)
    print('\nSpectral Parameters::')
    for p, r in zip(params, root_lists):
        print(p, r)
        
    print('\nComputing Ansatz::')
    ansatz = Bethe_Ansatz(N=N, V=1, W=0, params_and_E_options=(params, root_lists))
    angle_opts, c_opts, = ansatz.angle_options, ansatz.coeff_options
    print('\nAnsatz Info::')
    for angles, coeffs in zip(angle_opts, c_opts):
        print(coeffs)
        print(angles)

TEST_THRESHOLD = 1e-5

def test_N_7():
    #####
    N = 7
    W = .5
    V = .75

    # (1,6), (3,4), (5,2), (7,0); or greatest k to least k
    expected_C = list(reversed([3.40577e-3, -3.08911e-2, .18121, -.982953]))
    expected_theta = [3.13478, 3.20338, 9.78939]
    #####

    bethe_solver = Bethe_Equation_Solver(N=N, V=V, W=W)
    params, root_lists = bethe_solver.solve_bethe(verbose=True)
    print('\nSpectral Parameters::')
    print(params[1], root_lists[1])
    
    print('\nComputing Ansatz::')
    ansatz = Bethe_Ansatz(N=N, V=V, W=W, params_and_E_options=(params, root_lists), verbose=True)
    angle_opts, c_opts, = ansatz.angle_options, ansatz.coeff_options
    print('\nAnsatz Info::')
    print(c_opts[1])
    print(angle_opts[1])


    print('Expected:')
    print(expected_C)
    print(expected_theta)

    metrics = [ ('Coeff', zip(expected_C, c_opts[1])), ('Angle', zip(expected_theta, angle_opts[1])) ]
    for metric, result in metrics:
        print('\n' + metric + '::')
        diffs = [expected - actual for expected, actual in result]
        if all(abs(d) < TEST_THRESHOLD for d in diffs):
                print('Good!')
        else:
            print('Inaccurate output!')
    


if __name__ == '__main__':
    N = int(argv[1])
    verbose = len(argv) > 2 and argv[2] == '-v'
    main(N, verbose)
