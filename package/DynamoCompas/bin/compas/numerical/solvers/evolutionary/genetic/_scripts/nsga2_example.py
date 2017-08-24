import os,sys
from compas.numerical.solvers.evolutionary.genetic.algorithms.moga import MOGA

__author__     = ['Tomas Mendez Echenagucia <mtomas@ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'mtomas@ethz.ch'

GA = MOGA()

GA.fit_names             =  ['ZDT3function1','ZDT3function2']
GA.fit_type              =  [0,0]
GA.num_gen               =  200
GA.num_pop               =  50
GA.num_var               =  30
GA.mutation_probability  =  0.01
GA.start_from_gen        =  False

for i in range(GA.num_var):
    GA.boundaries[i]= [0.0,1.0]

GA.num_bin_dig  = []
for i in range(GA.num_var):
    GA.num_bin_dig.append(8)

GA.max_bin_dig = max(GA.num_bin_dig)

sumBinDigits = 0
for i in range(GA.num_var):
    sumBinDigits = sumBinDigits + GA.num_bin_dig[i]
GA.total_bin_dig = sumBinDigits

int_data = []

for name in GA.fit_names:
    fit =  __import__(name,'fitness')
    GA.fit_functions.append(fit.fitness)

GA.num_fit_func = len(GA.fit_names)
GA.output_path =  'out/'
if not os.path.exists(GA.output_path):
    os.makedirs(GA.output_path)

scaled = []
t_list = [1.0,0.0]
for j in range(len(t_list)):
    X = {}
    for i in range(GA.num_var):
        X[i] = t_list[j]
    scaled.append(X)

#GA.create_fixed_start_pop(scaled=scaled, binary=None)

GA.moga()

from compas.numerical.solvers.evolutionary.genetic.visualization.multi_objective_vis import MULTI_VIS
vis = MULTI_VIS()
vis.input_path = GA.output_path

filename = ''
for name in GA.fit_names:
    filename += name+'_'
filename +='.json'
#vis.generation = 1
vis.output_path = vis.input_path
#vis.scale = ((-0.05,1.05),(-0.05,1.05))
fit_list = ( (0,1),(1,0))
labels = ('A','B')
#vis.add_fixed_individuals(fit_list,labels)
vis.draw_objective_spaces(filename,number=False)



