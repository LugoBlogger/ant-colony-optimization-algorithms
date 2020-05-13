#-----------------------------------
# filename: ACO_TSACO_TSP.py
# __author__: Henokh Lugo Hariyanto
# __date__: April 11th, 2020
# __email__: lugoblogger@gmail.com
#-----------------------------------
# Usage:
# single experiment
# $ python ACO_TSACO_TSP.py -i "tsp_case.csv" -m 4 -b 2 -r -0.5 -algo 'TSACO'
#
# plot 1 with ACO algorithm
# $ python ACO_TSACO_TSP.py -i "tsp_case.csv" -m 4 -b 2 -r 0.5 -N 100 -p1 1 -algo 'ACO'
#  
# plot 2 with TSACO algorithm
# $ python ACO_TSACO_TSP.py -i "tsp_case.csv" -m 4 -b 2 -r 0.5 -N 100 -p2 1 -algo 'TSACO'
# 
# plot 3
# $ python ACO_TSACO_TSP.py -i "tsp_case.csv" -m 4 -b 2 -r 0.5 -N 100 -p3 1
#
# Type the following to see all the description in the arguments
# $ python ACO_TSACO_TSP.py -h
#
# when we provide with lower diagonal matrix for distance matrix
# put _ldm in the end of its name file
#
#----------------------------------

import numpy as np
import sys
import time
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt



def TSP_solver(G_dense, beta, rho, m, N_max_iter, algo, verbose=False):

    n = len(G_dense)

    mu = np.zeros_like(G_dense, dtype=float)
    mu[G_dense != 0] = 1./G_dense[G_dense != 0]
    
    # Initialize pheromone array
    tau0 = np.ones_like(G_dense)
    tau0_bool = np.array([[i == j for j in range(n)] for i in range(n)])
    tau0[tau0_bool] = np.zeros(n)

    max_Iter = 0

    for ell in range(N_max_iter):
        Tour_k = np.zeros((m, n+1), dtype=int)

        N_cities = np.arange(n, dtype=int)

        M_k = [N_cities[N_cities != 0] for _ in range(m)]

        tau = tau0.copy()

        p_k = np.array([[ tau[0, s] * mu[0, s]**beta for s in M_k[k]]
                        for k in range(m)])
        p_k = [p/p.sum() for p in p_k]

        cum_p_k = [[p[:i].sum() for i in range(len(p) + 1)] for p in p_k]
        cum_p_k = [[ [cum_p[i], cum_p[i+1]] for i in range(len(cum_p)-1)]
                   for cum_p in cum_p_k]


        for idx in range(n-1):
            r = np.random.rand(m)

            visited_city = [[lb <= r[k] < ub for lb, ub in cum_p]
                            for k, cum_p in enumerate(cum_p_k)]
            visited_city = [M_k[k][visited_city[k]][0] for k in range(m)]
            Tour_k[:, idx+1] = visited_city

            if len(M_k[0]) == 1:
                break

            M_k = [ M_k[k][M_k[k] != visited_city[k]] for k in range(m)]

            p_k = np.array([ [tau[visited_city[k], s] * mu[visited_city[k], s]**beta
                              for s in M_k[k]] for k in range(m)])
            p_k = [p/p.sum() for p in p_k]

            cum_p_k = [ [p[:i].sum() for i in range(len(p) + 1)] for p in p_k]
            cum_p_k = [ [[cum_p[i], cum_p[i+1]] for i in range(len(cum_p) - 1)]
                       for cum_p in cum_p_k]
        

        distance_k = np.array([ np.array([G_dense[Tour_k[k, i] , Tour_k[k, i+1]]
                                          for i in range(n)]).sum()
                               for k in range(m)])

        if verbose:
            print_tour(Tour_k)
        
        sys.stdout.write('\riteration: {:6d}, dist_min: {:5d}'.format(ell+1, distance_k.min()))
        sys.stdout.flush()

        best_distance = distance_k.min()
        best_Tour = Tour_k[best_distance == distance_k]
        best_ant = np.arange(m, dtype=int)[best_distance == distance_k]

        worst_distance = distance_k.max()
        worst_Tour = Tour_k[worst_distance == distance_k]
        worst_ant = np.arange(m, dtype=int)[worst_distance == distance_k]

        worst_Tour = ["-".join(["{:d}".format(path) for path in worst_Tour_i])
                      for worst_Tour_i in worst_Tour]


        # Evaporation of pheromone
        tau0 = (1. - rho)*tau

        # Deposition of pheromone
        if algo == 'ACO':
            for Tour, distance in zip(Tour_k, distance_k):
                for i in range(n):
                    tau0[Tour[i], Tour[i+1]] += 1./distance
                    tau0[Tour[i+1], Tour[i]] += 1./distance
        elif algo == 'TSACO':
            for Tour, distance in zip(Tour_k, distance_k):
                for i in range(n):
                    tau0[Tour[i], Tour[i+1]] += (rho/distance) * (worst_distance - distance)/float(best_distance)
                    tau0[Tour[i+1], Tour[i]] += tau0[Tour[i], Tour[i+1]]         
        else:
            print("Please set ACO or TSACO in -algo.")

        if len(best_Tour) == m:
            max_Iter = ell + 1
            break

    return max_Iter, best_distance, best_Tour[0]


def print_tour(Tour_k):
    Tour = Tour_k.copy()
    print("\n\nTour_k")
    for k, Tour_k in enumerate(Tour):
        print("Ant {:d}: {:}".format(k, Tour_k))


    return None


def single_experiment(variables):

    G_dense = variables[0]
    m = variables[1]
    beta = variables[2]
    rho = variables[3]
    N_max_iter = variables[4]
    algo = variables[6]


    start_time = time.perf_counter()
    iteration, dist_min, best_Tour = TSP_solver(G_dense, beta, rho, m, N_max_iter, algo, verbose=True)
    print("\nBest tour:", best_Tour)
    print("\nApproximated computational time: {:.4g} s".format(time.perf_counter() - start_time))

    return None


def visualize(variables):

    G_dense = variables[0]
    m = variables[1]
    beta = variables[2]
    rho = variables[3]
    N_max_iter = variables[4]
    plot_ants = variables[5]
    algo = variables[6]

    n = len(G_dense)

    if sum(plot_ants) == 1 or sum(plot_ants) == 0:
        plot1 = plot_ants[0]
        plot2 = plot_ants[1]
        plot3 = plot_ants[2]

        if sum(plot_ants) == 0:
            plot1 = True

        # N experiments
        if plot1:
            N_experiment = 100
            print("Number of experiments:", N_experiment)

            iter_arr = np.zeros(N_experiment, dtype=int)
            dist_arr = np.zeros(N_experiment, dtype=int)
            tour_arr = np.zeros((N_experiment, n+1), dtype=int)
            delta_avg = 0.
            for i in range(N_experiment):
                delta = time.perf_counter()
                iter_arr[i], dist_arr[i], tour_arr[i,:] = TSP_solver(G_dense, beta, rho, m, N_max_iter, algo)
                delta = time.perf_counter() - delta
                delta_avg += delta

            N_count_best = np.sum(dist_arr.min() == dist_arr)
            print("\nGlobal distance minimum:", dist_arr.min())
            print("Global best tour:", tour_arr[dist_arr.argmin(),:])
            print("Ratio:", N_count_best/float(N_experiment))
            print("Average computation on each experiment: {:4g} s".format(delta_avg/N_experiment)) 


            fig, ax = plt.subplots(dpi=100)
            x_experiment = np.arange(1, N_experiment+1, dtype=int)
            ax.plot(x_experiment, dist_arr)
            ax.set_xlabel("Experiment #-th")
            ax.set_ylabel("local optimal solution")
            ax.set_title("{:s}: m={:d}, beta={:g}, n={:d}, rho={:g}".format(algo, m, beta, n, rho)
                         + "\nglobal_opt_sol = {:d}  ({:2g}%)".format(int(dist_arr.min()),
                                                                     N_count_best/float(N_experiment) * 100))
            plt.tight_layout()
            plt.show() 


        # Experiment on number of ants
        if plot2:
            N_experiment = 100
            print("Number of experiments:", N_experiment)

            fig, ax = plt.subplots(nrows=2, ncols=1, dpi=100, figsize=(6,6))
            m_arr = [4, 8, 12, 16, 20, 24]              # you can modify this line

            time_spend_avg = np.zeros(len(m_arr))
            for j, m_arr_i in enumerate(m_arr):
                print("m:", m_arr_i)
                
                iter_arr = np.zeros(N_experiment, dtype=int)
                dist_arr = np.zeros(N_experiment, dtype=int)
                tour_arr = np.zeros((N_experiment, n+1), dtype=int)
                delta_avg = 0
                for i in range(N_experiment):
                    delta = time.perf_counter()
                    iter_arr[i], dist_arr[i], tour_arr[i,:] = TSP_solver(G_dense, beta, rho, m_arr_i, N_max_iter, algo)
                    delta = time.perf_counter() - delta
                    delta_avg += delta
                    
                time_spend_avg[j] = delta_avg/N_experiment
                N_count_best = np.sum(dist_arr.min() == dist_arr)
                print("\nGlobal distance minimum:", dist_arr.min())
                print("Global best tour:", tour_arr[dist_arr.argmin(),:])
                print("Ratio: {:2g}".format(N_count_best/float(N_experiment)))
                print("Average computation on each experiment: {:4g} s\n".format(delta_avg/N_experiment)) 

                x_experiment = np.arange(1, N_experiment+1, dtype=int)
                ax[0].plot(x_experiment, dist_arr, alpha=1. - 0.2*j, label="m={:d}".format(m_arr_i))
                
            ax[0].set_xlabel("experiment #-th")
            ax[0].set_ylabel("local optimal solution")
            ax[0].set_title("{:s}: beta={:g}, n={:d}, rho={:g}".format(algo, beta, n, rho))
            ax[0].legend(loc='upper left', bbox_to_anchor=[1.01, 1], borderaxespad=0, frameon=False)


            ax[1].plot(m_arr, time_spend_avg)
            ax[1].set_xlabel("number of ants")
            ax[1].set_ylabel("time (s)")
            ax[1].set_title("Avg. computation time on each experiment")
            ax[1].set_xticks(m_arr)

            plt.subplots_adjust(hspace=.5)
            plt.tight_layout()
            plt.show()

        
        # Comparing ACO and TSACO
        if plot3:
            N_experiment = 100
            print("Number of experiments:", N_experiment)
            
            y_iter_ACO = np.zeros(N_experiment, dtype=int)
            dist_min_ACO = np.zeros(N_experiment, dtype=int)
            tour_arr_ACO = np.zeros((N_experiment, n+1), dtype=int)
            
            y_iter_TSACO = np.zeros(N_experiment, dtype=int)
            dist_min_TSACO = np.zeros(N_experiment, dtype=int)
            tour_arr_TSACO = np.zeros((N_experiment, n+1), dtype=int)

            start_compute = time.perf_counter()
            print("ACO: ")
            for i in range(N_experiment):
                y_iter_ACO[i], dist_min_ACO[i], tour_arr_ACO[i,:] = TSP_solver(G_dense, beta, rho, m, N_max_iter, 'ACO')
            N_count_best_ACO = np.sum(dist_min_ACO.min() == dist_min_ACO)
            print("\n   Global best dist:", dist_min_ACO.min())
            print("Global best tour:", tour_arr_ACO[dist_min_ACO.argmin(),:])
            print("Approximation computational time: {:.4g} s".format(time.perf_counter() - start_compute))

            start_compute = time.perf_counter()
            print("\nTSACO: ")
            for i in range(N_experiment):
                y_iter_TSACO[i], dist_min_TSACO[i], tour_arr_TSACO[i,:] = TSP_solver(G_dense, beta, rho, m, N_max_iter, 'TSACO')
            N_count_best_TSACO = np.sum(dist_min_TSACO.min() == dist_min_TSACO)
            print("\n   Global best dist:", dist_min_TSACO.min())
            print("Global best tour:", tour_arr_TSACO[dist_min_TSACO.argmin(),:])
            print("Approximation computational time: {:.4g} s".format(time.perf_counter() - start_compute))


            fig, ax = plt.subplots(dpi=100, figsize=(13,6))
            x_experiment = np.arange(1, N_experiment+1)

            ax.plot(x_experiment, y_iter_ACO, label="ACO: {:d} ({:2g}%)".format(dist_min_ACO.min(), 
                                                                                (N_count_best_ACO/N_experiment)*100), 
                    alpha=0.8)
            ax.plot(x_experiment, y_iter_TSACO, label="TSACO {:d} ({:2g}%)".format(dist_min_TSACO.min(), 
                                                                                (N_count_best_TSACO/N_experiment)*100),
                    alpha=0.5)

            ax.set_xlabel("Experiment #-th")
            ax.set_ylabel("Number of iteration \nafter all ants move in the same paths")
            ax.set_title("m={:d}, beta={:g}, n={:d}, rho={:g}".format(m, beta, n, rho)\
                         + ", global_opt_sol={:d}".format(min(dist_min_ACO.min(), 
                                                              dist_min_TSACO.min())))
            plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.5, frameon=False)
            plt.tight_layout() 
            #figManager = plt.get_current_fig_manager()
            #figManager.window.showMaximized()
            plt.show() 


    else:
        print("Please plot only once for each execution.")

    return None


def to_variable(args):

    input_file = args.input_file

    with open(input_file) as f:
        read_data = f.readlines()

    G_dense = [line.strip('\n') for line in read_data]
    G_dense = [line.split(',') for line in G_dense]
    G_dense = [[int(r.strip()) for r in row] for row in G_dense]
   
    if input_file[-7:-4] == 'ldm':    # .csv only provided lower diagonal matrix
        G_dense_size = len(G_dense)
        G_dense_low = np.array([row + [0]*(G_dense_size - len(row)) for row in G_dense])
        G_dense_up = np.array([list(x) for x in zip(*G_dense_low)])
        G_dense = G_dense_low + G_dense_up

    G_dense = np.array(G_dense)
    print(G_dense)

    m = args.m
    beta = args.beta
    rho = args.rho
    N_max_filter = args.N_max_iter
    plot_arr = [args.plot1, args.plot2, args.plot3]
    algo = args.algo
    

    return [G_dense, m, beta, rho, N_max_filter, plot_arr, algo]


def read_input():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i',  type=str,
        default='tsp_case.csv', help='file.csv of the graph', dest='input_file')
    parser.add_argument('-m',  type=int,
        default=4, help='Number of ants', dest='m')
    parser.add_argument('-b',  type=int,
        default=2, help='Relative importance of heuristic distance',
        dest='beta')
    parser.add_argument('-r', type=float,
        default=0.5, help='Pheromone evaporation rate', dest='rho')
    parser.add_argument('-N', type=int,
        default=100, help='Maximum number of cycle of ants route',
        dest='N_max_iter')
    parser.add_argument('-p1', type=bool,
        default=False, help='Make a plot of local optimal solution\
                in 100 experiments', dest='plot1')
    parser.add_argument('-p2', type=bool,
        default=False, help='Make a plot of local optimal solution\
                and computational time for various number of ants',
        dest='plot2')
    parser.add_argument('-p3', type=bool,
        default=False, help='Make a plot to compare ACO and TSACO',
        dest='plot3')
    parser.add_argument('-algo', type=str,
        default='ACO', help='Select the algorithm between ACO or TSACO.\
        When we set -p3 1, we select both.', dest='algo')

    args = parser.parse_args()

    return args


def print_args(args):

    input_file = args.input_file
    
    m = args.m
    beta = args.beta
    rho = args.rho
    N_max_iter = args.N_max_iter
    
    plot1 = args.plot1
    plot2 = args.plot2
    plot3 = args.plot3
    algo = args.algo

    print("input_file:", input_file, type(input_file))
    print("m:", m, type(m))
    print("beta:", beta, type(beta))
    print("rho:", rho, type(rho))
    print("N_max_iter:", N_max_iter, type(N_max_iter))
    print("plot1:", plot1, type(plot1))
    print("plot2:", plot2, type(plot2))
    print("plot3:", plot3, type(plot3))
    print("algo:", algo, type(algo))

    return None



if __name__ == '__main__':

    #print(mpl.get_backend())
    args = read_input()
    #print_args(args)

    if len(sys.argv[1:]) == 10:
        variables = to_variable(args)
        single_experiment(variables)
    elif len(sys.argv[1:]) > 10:
        variables = to_variable(args)
        visualize(variables)
    else:
        print("Please provide with sufficient number of arguments")
