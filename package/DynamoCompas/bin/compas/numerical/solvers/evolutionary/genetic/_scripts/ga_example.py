from __future__ import print_function

import os

from compas.numerical.solvers.evolutionary.genetic.algorithms.ga import GA
from compas.numerical.solvers.evolutionary.genetic.visualization.ga_visualization import GA_VIS


__author__     = ['Tomas Mendez Echenagucia <mtomas@ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'mtomas@ethz.ch'


def foo(X, **kwargs):
    a = X.values()
    fit = sum(a)
    return fit


GA = GA()

GA.fit_name              = 'fitness1'
GA.fit_type              = 'min'
GA.num_gen               = 2000
GA.num_pop               = 10
GA.num_pop_init          = None
GA.num_gen_init_pop      = None
GA.num_elite             = 2
GA.num_var               = 30
GA.mutation_probability  = 0.005
GA.start_from_gen        = False
GA.min_fit               = GA.num_var * 0.01 + 0.001


for i in range(GA.num_var):
    GA.boundaries[i] = [0.01, 1]

GA.num_bin_dig = []
for i in range(GA.num_var):
    GA.num_bin_dig.append(4)

GA.max_bin_dig = max(GA.num_bin_dig)

sumBinDigits = 0
for i in range(GA.num_var):
    sumBinDigits = sumBinDigits + GA.num_bin_dig[i]
GA.total_bin_dig = sumBinDigits

int_data = []

GA.fit_function = foo

GA.num_fit_func = len(GA.fit_name)
GA.output_path = 'out/'
GA.input_path = GA.output_path

if not os.path.exists(GA.output_path):
    os.makedirs(GA.output_path)

GA.ga()

print('GA.best_individual_index', GA.best_individual_index)
print('best fit value', GA.current_pop['fit_value'][GA.best_individual_index])
print('best individual', GA.current_pop['decoded'][GA.best_individual_index])

vis = GA_VIS()
vis.input_path = GA.output_path
vis.output_path = vis.input_path
vis.conversion_function = None
vis.start_from_gen = 0

vis.draw_ga_evolution(make_pdf=False, show_plot=True)
