# ------------------------------------
#  TSACOalgo.py
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


def generate_x_cp(x_cp, i, tau, N_permit, x_worst_node):
    # This function will generate array with cumulative probabilty
    # and exclude an element whose objective function is the worst (max f).
    # We use the idea by putting negative value on (p_min, pmax).
    # This will exclude automatically roullete-wheel selection process.
    #
    # The following steps are the hierarchical on how to do branching:
    # 1) check whether the element in x_cp is an exclusion or not
    # 2) check the idx is equal to x_worst_node[0] or not
    # 3) check the carriage temp_upper is less than zero not
    
    N_tau = len(x_cp)
    x_cp_new = [0]*N_tau
    temp_upper = -1.   # this upper will be carried on through all the elements of of x_cp
    
    # first element
    if x_cp[0][0] < 0:
        x_cp_new[0] = (-2., -1.)
    else:
        if x_worst_node[0] == 1:
            x_cp_new[0] = (-2., -1.)
        else:
            upper = p_transition(i, 1, tau, N_permit)
            x_cp_new[0] = (0., upper)
            temp_upper = upper
        

    # second element to the (N_tau - 1)th element
    for idx in range(2, N_tau):
        if x_cp[idx-1][0] < 0:
            x_cp_new[idx-1] = (-2., -1.)
        else:
            if idx == x_worst_node[0]:
                x_cp_new[idx-1] = (-2., -1.)
            else:
                upper = p_transition(i, idx, tau, N_permit)
                if temp_upper < 0:
                    x_cp_new[idx-1] = (0., upper)
                    temp_upper = upper
                else:
                    x_cp_new[idx-1] = (temp_upper, temp_upper + upper)
                    temp_upper += upper
        
    
    # last element
    if x_cp[-1][0] < 0:
        x_cp_new[N_tau-1] = (-2., -1.)
    else:
        if x_worst_node[0] == N_tau:
            x_cp_new[N_tau-1] = (-2., -1.)
        else:
            upper = p_transition(i, N_tau, tau, N_permit)
            if temp_upper < 0:
                x_cp_new[N_tau-1] = (0., upper)
            else:
                x_cp_new[N_tau-1] = (temp_upper, 1)

                
    # normalize the last element which are not deleted should have the maximum 1
    for idx in range(N_tau):
        if x_cp_new[(N_tau-1)-idx][1] > 0:
            lower = x_cp_new[(N_tau-1)-idx][0]
            x_cp_new[(N_tau-1)-idx] = (lower, 1.)
            break
        
                
    
    return x_cp_new


def TSACO_algo(f, x_min, x_max, N_ant, N_permit, N_max_iter=10, test=True, seed=True, 
             verbose=True):
    ### algorithm to perform Ant Colony Optimization
    ### based on the book of Singiresu S. Rao,  
    ### "Engineering Optimization - Theory and Practice (2009, Wiley)"
    ### This is the modified version of ACO algorithm where 
    ### the path whose objective function is the worst will be skipped
    
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
        if i_iter == 0:
            x_cp = [cumulative_prob(1, j, tau, N_permit) for j in range(1, N_permit+1)]
        else:
            x_cp = generate_x_cp(x_cp, 1, tau, N_permit, x_worst_node)
        
        #print('x_cp: ', x_cp)
        
        ### For testing purpose with fixed r ###############
        if test:
            if i_iter == 0:
                r = np.array([0.5418, 0.2426, 0.3957, 0.5748])
            elif i_iter == 1:
                r = np.array([0.2645, 0.3567, 0.3697, 0.1536])   
            elif i_iter == 2:
                r = np.array([0.7187, 0.8904, 0.6076, 0.3616])
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
        
        #print('r: ', r)
            
        # determine next path
        next_path_node = np.array([ x_node[0, [cp_min < ri < cp_max for cp_min, cp_max in x_cp]] for ri in r])
        next_path_node = next_path_node.T + 1
        #print('next_path_node: ', next_path_node)
        
        next_path_val = np.array([x[0, j-1] for j in next_path_node[0]])
        #print('next_path_val: ', next_path_val)
        
        # compute objective function
        obj_func = f(next_path_val)
        #print('obj_func: ', obj_func)
        

        #---------------------------------------------------------
        # best and worst paths: x_best, f_best, x_worst, f_worst
        #---------------------------------------------------------

        bool_argmin = np.abs(obj_func.min() - obj_func) < tol
        x_best_node = next_path_node[0, bool_argmin]
        #print('x_best_node: ', x_best_node)
        
        x_best_val = next_path_val[bool_argmin]
        #print('x_best_val: ', x_best_val)
        
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

        tau_new[0, mask_x_best_node] = tau[0, mask_x_best_node] + sum_Delta_tau_best_k
        tau_new[0, mask_x_non_best_node] = (1 - rho) * tau[0, mask_x_non_best_node]  
        
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
            print('tau_new: ', tau_new)
            print('p_ij: ', [tau_ij/np.sum(tau_new[0])  for tau_ij in tau_new[0]])
        
        
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
         TSACO_algo(f, x_min, x_max, N_ant, N_permit, 
            N_max_iter=N_max_iter, test=test, seed=seed, verbose=False)

    print('optimal solution, x: ', optimal_sol)
    print('iter_max: ', iter_max)
    print('total time (approximately): {:.2g} s'.format(total_time ))