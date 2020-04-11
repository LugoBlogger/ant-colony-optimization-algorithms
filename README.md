# Ant Colony Optimization algorithms

An exploration of ACO-algorithm with its visualization(TBA). 

There are two types of solvers:

1. `ACOalgo.py` and `TSACOalgo.py` &ndash; These algorithms follow all the steps in 
Singiresu S. Rao - Engineering Optimization - Theory and Practice (2009, Wiley), section 13.5. They can only solve optimization problem involving one-variable function.
2. `ACO_TSACO_TSP.py &ndash; a travelling salesman problem solver for `n` cities. Up to six cities will give global optimal solution. Performing several trials, it will return the global optimal solution. Please use the argument `-algo 'TSACO` to use the improved version of ACO algorithn. Some useful literatures related to this program:
   - *Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem* by M. Dorigo and L. M. Gambardella.
   - *Ant Colony Optimization* by M. Dorigo and T. Stützle.
   - *ANTabu &ndash; enhanced version LIL-99-1* by O. Roux, C. Fonlupt, E.-G. Talbi, and D. Robilliard.
