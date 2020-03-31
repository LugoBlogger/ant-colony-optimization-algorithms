# ------------------------------------
#  ACOalgo.py
#  __author: Henokh Lugo Hariyanto
#  __Date created: March 31st, 2020
#  __email: lugoblogger@gmail.com
# ------------------------------------

import numpy as np
import time
import sys


def p_transition(i, j, tau, N_permit):
    ### function to compute probability of state transtion
    
    return tau[i-1, j-1] / np.sum([tau[i-1, p-1] for p in range(1, N_permit+1)])


def cumulative_prob(i, j, tau, N_permit):
    p_ij = p_transition(i, j, tau, N_permit)
    if j == 1:
        lower = 0.
        upper = lower + p_ij
    elif j == N_permit:
        lower = np.sum([p_transition(i, jPrev, tau, N_permit) for jPrev in range(1, j)])
        upper = 1.
    else:
        lower = np.sum([p_transition(i, jPrev, tau, N_permit) for jPrev in range(1, j)])
        upper = lower + p_ij
        
    return (lower, upper)


def ACO_algo(f, x_min, x_max, N_ant, N_permit, N_max_iter=10, test=True, seed=True, 
             verbose=True):
    ### algorithm to perform Ant Colony Optimization
    ### based on the book of Singiresu S. Rao,  
    ### "Engineering Optimization - Theory and Practice (2009, Wiley)"
    
    start_compute = time.time()
    
    tol = 1e-14
    n_design = 1
    rho = 0.5      # evaporation rate
    zeta = 2.      # control the scale of the global updating of
                   # the pheromone
    
    optimal_sol = 0.
    
    x_node = np.arange(N_permit, dtype=int)
    x_node = np.broadcast_to(x_node, (n_design, N_permit)) 
    #print('x_node: ', x_node)
    
    # generate permissible nodes, x_ij
    x = np.linspace(x_min, x_max, N_permit)
    x = np.broadcast_to(x, (n_design, N_permit))
    #print('x: ', x)

    # initialize pheromone array, t^(1)_ij
    tau = np.ones((n_design, N_permit))
    #print('tau: ', tau)
    
    for i_iter in range(N_max_iter):
    
        # generate cumulative probability ranges
        x_cp = [cumulative_prob(1, j, tau, N_permit) for j in range(1, N_permit+1)]
        
        
        ### For testing purpose with fixed r ###############
        if test:
            if i_iter == 0:
                r = np.array([0.3122, 0.8701, 0.4729, 0.6190])
            elif i_iter == 1:
                r = np.array([0.3688, 0.8577, 0.0706, 0.5791])   # there is a difference in textbook
            else:
                np.random.seed(2020 + i_iter)
                r = np.random.rand(N_ant)
        ####################################################
        else:
            # initialize random label to each ant
            if seed:
                np.random.seed(2020 +  + i_iter)    # for testing purpose with random r
                r = np.random.rand(N_ant)
            else:
                r = np.random.rand(N_ant)
          
            
        # determine next path
        next_path_node = np.array([ x_node[0, [cp_min < ri < cp_max for cp_min, cp_max in x_cp]] for ri in r])
        next_path_node = next_path_node.T + 1
        
        next_path_val = np.array([x[0, j-1] for j in next_path_node[0]])
        
        # compute objective function
        obj_func = f(next_path_val)
        

        #---------------------------------------------------------
        # best and worst paths: x_best, f_best, x_worst, f_worst
        #---------------------------------------------------------

        bool_argmin = np.abs(obj_func.min() - obj_func) < tol
        x_best_node = next_path_node[0, bool_argmin]
        x_best_val = next_path_val[bool_argmin]
        
        f_best_ant = np.arange(N_ant, dtype=int)[bool_argmin] + 1
        f_best_val = obj_func[bool_argmin]
        
        bool_argmax = np.abs(obj_func.max() - obj_func) < tol
        x_worst_node = next_path_node[0, bool_argmax]
        x_worst_val = next_path_val[bool_argmax]
        
        f_worst_ant = np.arange(N_ant, dtype=int)[bool_argmax] + 1
        f_worst_val = obj_func[bool_argmax]
       
        # pheromone deposited by the best ant k
        #sum_Delta_tau_best_k = len(f_best_val)*zeta*f_best_val[0]/f_worst_val[0]
        
            # solution when f_best and f_worst have different sign
        sum_Delta_tau_best_k = len(f_best_val)*zeta*np.abs(f_best_val[0]/f_worst_val[0])  
        

        #--------------------------------------------------
        # update pheromone array:
        #  (+) pheromone amount of the previous iteration 
        #      left after evaporation
        #  (+) we don't need to add sum_Delta_tau_best_k 
        #   to non_best paths, see Eq. (13.37)
        #--------------------------------------------------

        tau_new = np.zeros_like(tau)

        mask = np.arange(N_permit, dtype=int) + 1
        mask_x_best_node = mask == x_best_node[0]   # only need one
        mask_x_non_best_node = mask != x_best_node[0]

        print
        tau_new[0, mask_x_best_node] = tau[0, mask_x_best_node] + sum_Delta_tau_best_k
        tau_new[0, mask_x_non_best_node] = (1 - rho) * tau[0, mask_x_non_best_node] 
        #print('tau_new: ', tau_new)
        
        tau = tau_new.copy()
        
        # verbosity:
        if verbose:
            print('x_cp: ', x_cp)
            print('r: ', r)
            print('next_path_node: ', next_path_node)
            print('next_path_val: ', next_path_val)
            print('obj_func: ', obj_func)

            print('x_best_node: ', x_best_node)
            print('x_best_val: ', x_best_val)
            print('f_best_ant: ', f_best_ant)
            print('f_best_val: ', f_best_val)

            print('x_worst_node: ', x_worst_node)
            print('x_worst_val: ', x_worst_val)
            print('f_worst_ant: ', f_worst_ant)
            print('f_worst_val: ', f_worst_val)
        
            print('sum_Delta_tau_best_k: ', sum_Delta_tau_best_k)
        
        
        if len(x_best_node) == N_ant:   # all ants choose the same best path
            #print('Number of iteration: ', i_iter + 1)
            
            optimal_sol = x[0, x_best_node[0] - 1]
            #print('optimal solution, x:', optimal_sol)
            
            break
    
    total_time = time.time() - start_compute
    
    return optimal_sol, i_iter + 1, total_time



if __name__ == '__main__':
        
    #--- 1st test -----------------------------
    #f = lambda x: x**2 - 2.*x - 11.          
    #test = True
    #seed = True
    #------------------------------------------

    #--- 2nd test -----------------------------
    #f = lambda x: x**5 - 5.*x**3 - 20.*x + 5.
    #test = False
    #seed = False
    #------------------------------------------

    #-- Put your function in here -------------
    f = lambda x: x**2 - 2.*x - 11. 
    test = False 
    seed = False       
    # -----------------------------------------
    
    N_ant = 4

    x_min = 0.
    x_max = 3.

    N_permit = 7
    N_max_iter = 20

    optimal_sol, iter_max, total_time =\
         ACO_algo(f, x_min, x_max, N_ant, N_permit, 
            N_max_iter=N_max_iter, test=test, seed=seed, verbose=False)

    print('optimal solution, x: ', optimal_sol)
    print('iter_max: ', iter_max)
    print('total time (approximately): {:.2g} s'.format(total_time ))