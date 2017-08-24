from __future__ import print_function

import re
import random
import json

__author__     = ['Tomas Mendez Echenagucia <mtomas@ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'mtomas@ethz.ch'


class MOGA:
    """This class contains a binary coded, multiple objective genetic algorithm called
    NSGA-II (ref to K. Deb). NSGA-II uses the concept of non-domination, or Pareto-domination to
    classify solutions and optimize as a genetic algorith. NSGA-II also employs a crowding distance
    operator especially designed to distribute individuals in the population allong the Pareto
    front, and this avoid crowding in small areas. The main function is ``MOGA.moga``, calling
    this starts the multi objective optimization process.
    """

    def __init__(self):
        """
        Populations are kept in dictionaries with the following data structure_ansys:
        pop = {'binary':binary_dict,'decoded':decoded_dict,'scaled':scaled_dict,
                'fit_values':fit_values_dict,'pf':pf_dict}
        binary_dict     = {individual index: {variable index : binary list}}
        decoded_dict    = {individual index: {variable index : decoded variable}}
        scaled_dict     = {individual index: {variable index : scaled variable}}
        fit_values_dict = {individual index: {fit function index : fitness value}}
        pf_dict         = {individual index: pf number}
        """
        self.fit_functions = []
        self.num_var = 0
        self.num_pop = 0
        self.num_gen = 0
        self.boundaries   = {}
        self.num_bin_dig = 0
        self.mutation_probability = 0
        self.fit_names = []
        self.fit_type = []
        self.start_from_gen = False
        self.max_bin_dig = 0
        self.total_bin_dig = 0
        self.num_fit_func = 0
        self.output_path = []
        self.additional_data = {}
        self.parent_combined_dict = {}
        self.parent_pop   = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_values': {}, 'pf': {}}
        self.current_pop  = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_values': {}}
        self.combined_pop = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_values': {}}
        self.new_pop_cd = []
        self.fixed_start_pop = None  # {'binary':{},'decoded':{},'scaled':{}}

    def moga(self):
        """ This is the main optimization function, this function permorms the multi objective
        GA optimization, performing all genetic operators.
        """
        self.write_moga_json_file()
        if self.start_from_gen:
            self.parent_pop = self.get_pop_from_pf_file()
            start_gen_number = self.start_from_gen + 1
        else:
            start_gen_number = 0
            self.parent_pop['binary'] = self.generate_random_bin_pop()
            self.parent_pop['decoded'] = self.decode_binary_pop(self.parent_pop['binary'])
            self.parent_pop['scaled'] = self.scale_population(self.parent_pop['decoded'])

            if self.fixed_start_pop:
                for i in range(self.fixed_start_pop['num_pop']):
                    print('fixed start individual', i)
                    print(self.fixed_start_pop['binary'][i])
                    self.parent_pop['binary'][i] = self.fixed_start_pop['binary'][i]
                    self.parent_pop['decoded'][i] = self.fixed_start_pop['decoded'][i]
                    print(self.fixed_start_pop['decoded'][i])
                    self.parent_pop['scaled'][i] = self.fixed_start_pop['scaled'][i]
                    print(self.fixed_start_pop['scaled'][i])
                    print('')

            for i in range(self.num_pop):
                self.parent_pop['fit_values'][i] = {}
                for j in range(self.num_fit_func):
                    fit_func = self.fit_functions[j]
                    self.parent_pop['fit_values'][i][j] = fit_func(self.parent_pop['scaled'][i], self.additional_data)

        self.current_pop['binary'] = self.generate_random_bin_pop()

        for generation in range(start_gen_number, self.num_gen):
            print('generation ', generation)

            self.current_pop['decoded'] = self.decode_binary_pop(self.current_pop['binary'])
            self.current_pop['scaled'] = self.scale_population(self.current_pop['decoded'])

            for i in range(self.num_pop):
                self.current_pop['fit_values'][i] = {}
                for j in range(self.num_fit_func):
                    fit_func = self.fit_functions[j]
                    self.current_pop['fit_values'][i][j] = fit_func(self.current_pop['scaled'][i], self.additional_data)

            self.combine_populations()
            self.non_dom_sort()

            for u in range(len(self.pareto_front_indices) - 1):
                self.extract_pareto_front(u)
                self.calculate_crowding_distance()

            self.crowding_distance_sorting()
            self.parent_reseting()
            self.write_out_file(generation)

            if generation < self.num_gen - 1:
                self.nsga_tournament()
                self.create_mating_pool()
                self.simple_crossover()
                self.random_mutation()
            else:
                print('end of MOGA')

    def write_out_file(self, generation):
        """This function writes a file containing all of the population data for
        the given ``generation``.

        Parameters
        ----------
        generation: int
            The generation to write the population data of.
        """
        filename  = 'generation ' + "%03d" % generation + '_pareto_front' + ".pareto"
        pf_file  = open(self.output_path + (str(filename)), "wb")
        pf_file.write('Generation \n')
        pf_file.write(str(generation) + '\n')
        pf_file.write('\n')

        pf_file.write('Number of individuals per generation\n')
        pf_file.write(str(self.num_pop))
        pf_file.write('\n')
        pf_file.write('\n')

        pf_file.write('Population scaled variables \n')
        for i in range(self.num_pop):
            pf_file.write(str(i) + ',')
            for f in range(self.num_var):
                pf_file.write(str(self.parent_pop['scaled'][i][f]))
                pf_file.write(',')
            pf_file.write('\n')
        pf_file.write('\n')

        pf_file.write('Population fitness values \n')
        for i in range(self.num_pop):
            pf_file.write(str(i) + ',')
            for f in range(self.num_fit_func):
                pf_file.write(str(self.parent_pop['fit_values'][i][f]))
                pf_file.write(',')
            pf_file.write('\n')
        pf_file.write('\n')

        pf_file.write('Population Pareto front indices \n')
        for i in range(self.num_pop):
            pf_file.write(str(i) + ',')
            pf_file.write(str(self.parent_pop['pf'][i]) + '\n')
        pf_file.write('\n')

        pf_file.write('\n')
        pf_file.close()

    def generate_random_bin_pop(self):
        """This function generates a random binary population

        Retunrs
        -------
        rendom_bin_pop: dict
            The generated random binary population dictionary.
        """
        random_bin_pop = {}
        for j in range(self.num_pop):
            random_bin_pop[j] = {}
            for i in range(self.num_var):
                chrom_list = []
                for u in range(self.num_bin_dig[i]):
                    chrom_list.append(random.randint(0, 1))
                random_bin_pop[j][i] = chrom_list

        return random_bin_pop

    def decode_binary_pop(self, bin_pop):
        """This function decodes the given binary population

        Parameters
        ----------
        bin_pop: dict
            The binary population to decode.

        Returns
        -------
        decoded_pop: dict
            The decoded population dictionary.
        """
        decoded_pop = {}
        for j in range(len(bin_pop)):
            decoded_pop[j] = {}
            for i in range(self.num_var):
                value = 0
                chrom = bin_pop[j][i]
                for u, gene in enumerate(chrom):
                    if gene == 1:
                        value = value + 2**u
                decoded_pop[j][i] = value
        return decoded_pop

    def scale_population(self, decoded_pop):
        """Scales the decoded population, variable values are scaled according to each
        of their bounds contained in ``MOGA.boundaries``.

        Parameters
        ----------
        decoded_pop: dict
            The decoded population dictionary.

        Returns
        -------
        scaled_pop: dict
            The scaled ppopulation dictionary.
        """
        scaled_pop = {}
        max_bin = []
        for q in range(self.num_var):
            max_bin_temp = 0
            for u in range(self.num_bin_dig[q]):
                max_bin_temp = max_bin_temp + 2**u
            max_bin.append(max_bin_temp)

        for j in range(len(decoded_pop)):
            scaled_pop[j] = {}
            for i in range(self.num_var):
                scaled_pop[j][i] = 1.0 / max_bin[i] * decoded_pop[j][i]

        for j in range(len(decoded_pop)):
            for i in range(self.num_var):
                bound = self.boundaries[i]
                scaled_pop[j][i] = bound[0] + (bound[1] - bound[0]) * scaled_pop[j][i]

        return scaled_pop

    def combine_populations(self):
        """This function combines the parent population with the current population
        to create a 2 x ``MOGA.num_pop`` long current population.
        """
        for i in range(self.num_pop):
            self.combined_pop['binary'][i] = self.parent_pop['binary'][i]
            self.combined_pop['binary'][i + self.num_pop] = self.current_pop['binary'][i]

            self.combined_pop['decoded'][i] = self.parent_pop['decoded'][i]
            self.combined_pop['decoded'][i + self.num_pop] = self.current_pop['decoded'][i]

            self.combined_pop['scaled'][i] = self.parent_pop['scaled'][i]
            self.combined_pop['scaled'][i + self.num_pop] = self.current_pop['scaled'][i]

            self.combined_pop['fit_values'][i] = self.parent_pop['fit_values'][i]
            self.combined_pop['fit_values'][i + self.num_pop] = self.current_pop['fit_values'][i]

    def non_dom_sort(self):
        """This function performs the non dominated sorting operator of the NSGA-II
        algorithm. It assigns each individual in the population a Pareto front level,
        according to their fitness values.
        """
        self.domination_count = {}
        self.dominated_set = []
        self.dominating_individuals = []
        self.pareto_front_indices = []
        self.pareto_front_individuals = []

        for i in range(self.num_pop * 2):
            self.domination_count[i] = 0

            for k in range(self.num_pop * 2):
                if i == k:
                    continue
                count_sup = 0
                count_inf = 0
                for j in range(self.num_fit_func):
                    if self.fit_type[j] == 0:
                        if self.combined_pop['fit_values'][i][j] < self.combined_pop['fit_values'][k][j]:
                            count_sup += 1
                        elif self.combined_pop['fit_values'][i][j] > self.combined_pop['fit_values'][k][j]:
                            count_inf += 1
                    elif self.fit_type[j] == 1:
                        if self.combined_pop['fit_values'][i][j] > self.combined_pop['fit_values'][k][j]:
                            count_sup += 1
                        elif self.combined_pop['fit_values'][i][j] < self.combined_pop['fit_values'][k][j]:
                            count_inf += 1

                if count_sup < 1 and count_inf >= 1:
                    self.domination_count[i] += 1

                elif count_sup >= 1 and count_inf < 1:
                    self.dominated_set.append(k)
                    self.dominating_individuals.append(i)

        pareto_front_number = 0
        self.pareto_front_indices.append(0)
        while len(self.pareto_front_individuals) < self.num_pop:
            index_count = 0
            for i in range(self.num_pop * 2):
                if self.domination_count[i] == 0:
                    self.pareto_front_individuals.append(i)
                    self.domination_count[i] -= 1
                    index_count += 1

            index = index_count + self.pareto_front_indices[pareto_front_number]
            self.pareto_front_indices.append(index)

            a  = self.pareto_front_indices[pareto_front_number]
            b  = self.pareto_front_indices[pareto_front_number + 1]

            for k in range(a, b):
                for h in range(len(self.dominating_individuals)):
                    if self.pareto_front_individuals[k] == self.dominating_individuals[h]:
                        if self.domination_count[self.dominated_set[h]] >= 0:
                            self.domination_count[self.dominated_set[h]] = self.domination_count[self.dominated_set[h]] - 1

            pareto_front_number += 1

    def extract_pareto_front(self, u):
        """Adds each new level of pareto front individuals to the ``MOGA.i_pareto_front`` list.
        """
        self.i_pareto_front = []
        for i in range(self.pareto_front_indices[u], self.pareto_front_indices[u + 1]):
            self.i_pareto_front.append(self.pareto_front_individuals[i])

    def calculate_crowding_distance(self):
        """This function calculates the crowding distance for all inividuals in the population.
        The crowding distance computes the volume of the hypercube that surrounds each individual
        and whose boundaries are determined by their closest neighbors in the objective space. The
        crowdng distance is used by NSGA-II to better distribute the population allong the Pareto
        front and avoid crowded areas, thus better representing the variety of solutions in the front.
        """
        self.num_i_pareto_front      = len(self.i_pareto_front)
        self.pf_values               = [0] * self.num_i_pareto_front
        self.crowding_distance       = [0] * self.num_i_pareto_front

        for i in range(self.num_fit_func):
            ind_fit_values_list = [self.combined_pop['fit_values'][key][i] for key in self.combined_pop['fit_values'].keys()]
            delta = max(ind_fit_values_list) - min(ind_fit_values_list)

            for k in range(self.num_i_pareto_front):
                self.pf_values[k] = (self.combined_pop['fit_values'][self.i_pareto_front[k]][i])

            if self.fit_type[i] == 1:
                self.sorted_indices = self.get_sorting_indices(self.pf_values, reverse=False)
            else:
                self.sorted_indices = self.get_sorting_indices(self.pf_values, reverse=False)

            self.crowding_distance[self.sorted_indices[0]] = float('inf')
            self.crowding_distance[self.sorted_indices[self.num_i_pareto_front - 1]] = float('inf')

            for j in range(1, self.num_i_pareto_front - 1):
                formula = (self.pf_values[self.sorted_indices[j + 1]] - self.pf_values[self.sorted_indices[j - 1]]) / delta
                self.crowding_distance[self.sorted_indices[j]] += formula

        for i in range(self.num_i_pareto_front):
            self.new_pop_cd.append(self.crowding_distance[i])

    def get_sorting_indices(self, l, reverse=False):
        """Reurns the indices that would sort a list of floats.

        Parameters
        ----------
        l: list
            The list of floats to be sorted.
        reverse: bool
            If true the sorting will be done from top to bottom.

        Returns
        -------
        sorting_index: list
            The list of indices that would sort the given list of floats.
        """
        sorting_index = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(l))]
        if reverse is True:
            sorting_index = list(reversed(sorting_index))
        return sorting_index

    def crowding_distance_sorting(self):
        """This function sorts the individuals in the population according
        to their crowding distance.
        """
        cd_sorted_last_pf_index = []
        sorted_last_pf_cd  = sorted(self.crowding_distance)
        sorted_last_pf_cd = list(reversed(sorted_last_pf_cd))
        sorting_index = self.get_sorting_indices(self.crowding_distance, reverse=True)

        for i in range(self.num_i_pareto_front):
            cd_sorted_last_pf_index.append(self.i_pareto_front[sorting_index[i]])

        self.new_pop_cd[len(self.new_pop_cd) - self.num_i_pareto_front:len(self.new_pop_cd)] = sorted_last_pf_cd[:]
        self.pareto_front_individuals[len(self.new_pop_cd) - self.num_i_pareto_front: len(self.new_pop_cd)] = cd_sorted_last_pf_index[:]

    def parent_reseting(self):
        """This function updates the patent population, selecting the individuals that are higher
        in the pareto front level, and have the largest crowding distance.
        """
        self.parent_pop['scaled'] = {}
        self.parent_pop['decoded'] = {}
        self.parent_combined_dict = {}

        for i in range(self.num_pop):
            self.parent_pop['binary'][i] = self.combined_pop['binary'][self.pareto_front_individuals[i]]
            self.parent_pop['fit_values'][i] = self.combined_pop['fit_values'][self.pareto_front_individuals[i]]
            self.parent_combined_dict[i] = self.pareto_front_individuals[i]

        self.parent_pop['decoded'] = self.decode_binary_pop(self.parent_pop['binary'])
        self.parent_pop['scaled']  = self.scale_population(self.parent_pop['decoded'])
        self.parent_pop['pf'] = self.make_pop_pf_dict()

    def nsga_tournament(self):
        """This function performs the tournament selection operator of the NSGA-II
        algorithm.
        """
        pf_indices_a                = [0] * self.num_pop
        pf_indices_b                = [0] * self.num_pop
        cd_b                        = [0] * self.num_pop
        self.mp_individual_indices  = [0] * self.num_pop

        temp_pf_individuals_a = []
        temp_pf_individuals_a[:]  = self.pareto_front_individuals[0:self.num_pop]
        temp_pf_individuals_b = random.sample(temp_pf_individuals_a, len(temp_pf_individuals_a))

        pf_individuals_2 = []
        indices = []

        cd_a = self.new_pop_cd

        for i in range(self.num_pop):
            while temp_pf_individuals_a[i] == temp_pf_individuals_b[0] and i != self.num_pop - 1:
                temp_pf_individuals_b = random.sample(temp_pf_individuals_b, len(temp_pf_individuals_b))

            pf_individuals_2.append(temp_pf_individuals_b[0])
            del temp_pf_individuals_b[0]
            # t emp_pf_individuals_b = np.delete(tempPFindividualsB,0,0)

        for j in range(len(self.pareto_front_indices) - 1):
            pf_indices_a[self.pareto_front_indices[j]: self.pareto_front_indices[j + 1]] = [j] * (self.pareto_front_indices[j + 1] - self.pareto_front_indices[j])

        for i in range(len(pf_individuals_2)):
            for u in range(len(temp_pf_individuals_a)):
                if pf_individuals_2[i] == temp_pf_individuals_a[u]:
                    indices.append(u)

        for k in range(self.num_pop):
            pf_indices_b[k] = pf_indices_a[indices[k]]

        for k in range(self.num_pop):
            cd_b[k] = cd_a[indices[k]]

        for j in range(self.num_pop):
            if pf_indices_a[j] > pf_indices_b[j]:
                self.mp_individual_indices[j] = pf_individuals_2[j]
            elif pf_indices_a[j] < pf_indices_b[j]:
                self.mp_individual_indices[j] = temp_pf_individuals_a[j]
            else:
                if cd_a[j] > cd_b[j]:
                    self.mp_individual_indices[j] = temp_pf_individuals_a[j]
                elif cd_a[j] < cd_b[j]:
                    self.mp_individual_indices[j] = pf_individuals_2[j]
                else:
                    self.mp_individual_indices[j] = temp_pf_individuals_a[j]

        self.pareto_front_indices       = []
        self.sorted_crowding_distance   = []
        self.new_pop_cd                 = []
        self.dominated_set              = []
        self.dominating_individuals     = []
        self.pareto_front_individuals   = []
        self.new_pop_cd                 = []

    def create_mating_pool(self):
        """Creates two lists of cromosomes to be used by the crossover operator.
        """
        self.mating_pool_a = []
        self.mating_pool_b = []
        for i in range(self.num_pop / 2):
            chrom_a = []
            chrom_b = []
            for j in range(self.num_var):
                chrom_a += self.combined_pop['binary'][self.mp_individual_indices[i]][j]
                chrom_b += self.combined_pop['binary'][self.mp_individual_indices[i + (self.num_pop / 2)]][j]
            self.mating_pool_a.append(chrom_a)
            self.mating_pool_b.append(chrom_b)

    def simple_crossover(self):
        """Performs the simple crossover operator. Individuals in ``MOGA.mating_pool_a`` are
        combined with individuals in ``MOGA.mating_pool_b`` using a single, randomly selected
        crossover point.
        """
        self.current_pop = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_values': {}}

        for j in range(self.num_pop / 2):
            cross = random.randint(1, self.total_bin_dig - 1)
            a = self.mating_pool_a[j]
            b = self.mating_pool_b[j]
            c = a[:cross] + b[cross:]
            d = b[:cross] + c[cross:]

            self.current_pop['binary'][j] = {}
            self.current_pop['binary'][j + (self.num_pop / 2)] = {}
            for i in range(self.num_var):
                variable_a = c[:self.num_bin_dig[i]]
                del c[:self.num_bin_dig[i]]
                variable_b = d[:self.num_bin_dig[i]]
                del d[:self.num_bin_dig[i]]
                self.current_pop['binary'][j][i] = variable_a
                self.current_pop['binary'][j + (self.num_pop / 2)][i] = variable_b

    def random_mutation(self):
        """This mutation operator replaces a gene from 0 to 1 or viceversa
        with a probability of ``MOGA.mutation_probability``.
        """
        for i in range(self.num_pop):
            for j in range(self.num_var):
                for u in range(self.num_bin_dig[j]):
                    random_value = random.random()
                    if random_value < (self.mutation_probability):
                        if self.current_pop['binary'][i][j][u] == 0:
                            self.current_pop['binary'][i][j][u] = 1
                        else:
                            self.current_pop['binary'][i][j][u] = 0

    def get_pop_from_pf_file(self):
        """Reads the pareto front file corresponding to the ``MOGA.start_from_gen``
        generation and returns the saved population data. The pareto front file
        must be in ``GA.output_path``.

        Returns
        -------
        file_pop: dict
            The population dictionary contained in the file.
        """

        file_pop = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_values': {},
                    'pf': {}}
        filename  = 'generation ' + "%03d" % self.start_from_gen + '_pareto_front' + ".pareto"
        filename = self.output_path + filename
        pf_file = open(filename, 'r')
        lines = pf_file.readlines()
        pf_file.close()

        for i in range(self.num_pop):
            file_pop['scaled'][i] = {}
            file_pop['fit_values'][i] = {}

            line_scaled = lines[i + 7]
            line_fit = lines[i + 9 + self.num_pop]
            line_pf = lines[i + 11 + (self.num_pop * 2)]

            string_scaled = re.split(',', line_scaled)
            string_fit = re.split(',', line_fit)
            string_pf = re.split(',', line_pf)

            del string_scaled[-1]
            del string_scaled[0]
            del string_fit[-1]
            del string_fit[0]

            scaled = [float(j) for j in string_scaled]
            fit_values = [float(j) for j in string_fit]
            pf = int(string_pf[1])

            for j in range(len(scaled)):
                file_pop['scaled'][i][j] = scaled[j]

            for j in range(len(fit_values)):
                file_pop['fit_values'][i][j] = fit_values[j]

            file_pop['pf'][i] = pf

        file_pop['decoded'] = self.unscale_pop(file_pop['scaled'])
        file_pop['binary']  = self.code_decoded(file_pop['decoded'])
        return file_pop

    def code_decoded(self, decoded_pop):
        """Returns a binary coded population from a decoded population

        Parameters
        ----------
        decoded_pop: dict
        The decoded population dictionary to be coded

        Returns
        -------
        binary_pop: dict
            The binary population dictionary.
        """
        binary_pop = {}
        for i in range(len(decoded_pop)):
            binary_pop[i] = {}
            for j in range(self.num_var):
                bin_list = []
                temp_bin = bin(decoded_pop[i][j])[2:]
                temp_bin = temp_bin[::-1]
                digit_dif = self.num_bin_dig[j] - len(temp_bin)
                for h in range(digit_dif):
                    temp_bin = temp_bin + '0'
                for k in range(self.num_bin_dig[j]):
                    bin_list.append(int(temp_bin[k]))
                binary_pop[i][j] = bin_list
        return binary_pop

    def unscale_pop(self, scaled_pop):
        """Returns an unscaled population from a scaled one. The variable values are scaled
        from 0 to x, where x is the highest number described by the number of binary digits
        used to encode that variable. For example, if ``GA.num_bin_dig`` for a variable is 8, that
        variable will be scaled back from its bounds to its corresponding value from 0 to 255.

        Parameters
        ----------
        scaled_pop: dict
            the scaled population dictionary.

        Returns
        -------
        unscaled_pop: dict
            The unscaled population dictionary.
        """
        unscaled_pop = {}
        for i in range(len(scaled_pop)):
            unscaled_pop[i] = {}
            for j in range(self.num_var):
                bin_dig = self.num_bin_dig[j]
                bounds = self.boundaries[j]
                max_unscaled_value = self.get_max_value_from_bin_big(bin_dig)
                dom = abs(bounds[1] - bounds[0])
                value_s = scaled_pop[i][j]
                value = (value_s - bounds[0]) / float(dom)
                unscaled = int(value * max_unscaled_value)
                unscaled_pop[i][j] = unscaled

        return unscaled_pop

    def get_max_value_from_bin_big(self, bin_dig):
        """Returns the highest number described by a ``GA.bin_dig`` long binary number.

        Parameters
        ----------
        bin_dig: int
            The number of digits in the binary number.

        Returns
        -------
        value: int
            The highest number described by a ``GA.bin_dig`` long binary number.
        """
        binary = ''
        for i in range(bin_dig):
            binary += '1'
        value = 0
        for i in range(bin_dig):
            value = value + 2**i
        return value

    def make_pop_pf_dict(self):
        """This function returns a dictionary containing the pareto front level of
        all individuals in the population

        Returns
        -------
        pf_dict: dict
            The dictionary containing the pareto front level of all individuals.

        """
        pf_dict = {}
        for j in range(len(self.pareto_front_indices) - 1):
            pf_ind = self.pareto_front_individuals[self.pareto_front_indices[j]:self.pareto_front_indices[j + 1]]
            for i in range(self.num_pop):
                index = self.parent_combined_dict[i]
                if index in pf_ind:
                    pf_dict[i] = j

        return pf_dict

    def make_moga_input_data(self):
        """Returns a dictionary containing the most relavant genetic data present in the instance
        of ``MOGA``. This is the data required to restart a genetic optimization process or to
        launch a visualization using ``compas_ga.visualization.moga_visualization``.

        Returns
        -------
        data: dict
            A dictionary containing genetic data.
        """
        data = {'num_var': self.num_var,
                'num_pop': self.num_pop,
                'num_gen': self.num_gen,
                'boundaries': self.boundaries,
                'num_bin_dig': self.num_bin_dig,
                'mutation_probability': self.mutation_probability,
                'fit_names': self.fit_names,
                'fit_type': self.fit_type,
                'start_from_gen': self.start_from_gen,
                'max_bin_dig': self.max_bin_dig,
                'total_bin_dig': self.total_bin_dig,
                'num_fit_func': self.num_fit_func,
                'output_path': self.output_path,
                'fixed_start_pop': self.fixed_start_pop
                # 'additional_data':self.additional_data,
                }
        return data

    def make_gen_data(self):
        """Returns a dictionary containing the most relavant genetic data present in the instance
        of ``MOGA`` for the current generation only.

        Returns
        -------
        data: dict
            A dictionary containing genetic data.
        """

        data = {'parent_fit_values': self.parent_pop['fit_values'],
                'parent_scaled': self.parent_pop['scaled'],
                'parent_pf': self.parent_pop['pf']
                }
        return data

    def write_moga_json_file(self):
        """Writes a JSON file containing the most relevant data for MOGA optimization and
        visualization using ``compas_ga.visualization.moga_visualization``.
        """
        data = self.make_moga_input_data()
        filename = ''
        for name in self.fit_names:
            filename += name + '_'
        filename += '.json'
        with open(self.output_path + filename, 'wb+') as fh:
            json.dump(data, fh)

    def write_gen_json_file(self, generation):
        """Writes a JSON file containing the most relevant data for MOGA optimization and
        visualization using ``compas_ga.visualization.ga_visualization`` for the given
        generation ``generation``.

        Parameters:
        -----------
        generation:int
            The generation to write the JSON file for.
        """
        data = self.make_gen_data()
        filename = 'generation ' + "%03d" % generation + '_pareto_front' + ".json"
        fh = open(self.output_path + filename, 'wb+')
        json.dump(data, fh)

    def create_fixed_start_pop(self, scaled=None, binary=None):
        """This function creates a population to start the MOGA from a given scaled
        or binary populaiton. This function is used then the start of the MOGA
        should not be with a random population, but with a user defined population.
        The user may chose a binary or a scaled population, one of them must be
        given as a keyword argument. The fixed starting population is saved in
        ``MOGA.fixed_start_pop``. If this function is used, the ``MOGA.moga``
        function will automatically use the ``MOGA.fixed_start_pop`` instead of a
        random population.

        Parameters
        ----------
        scaled: dict
            The scaled population to start the MOGA process from.
        binary: dict
            The binary population to start the MOGA process from.
        """
        self.fixed_start_pop = {'binary': {}, 'decoded': {}, 'scaled': {}, 'num_pop': 0}

        if scaled:
            self.fixed_start_pop['num_pop'] = len(scaled)

            for i in range(self.fixed_start_pop['num_pop']):
                self.fixed_start_pop['scaled'][i] = scaled[i]

            self.fixed_start_pop['decoded'] = self.unscale_pop(self.fixed_start_pop['scaled'])
            self.fixed_start_pop['binary'] = self.code_decoded(self.fixed_start_pop['decoded'])

        if binary:
            self.fixed_start_pop['num_pop'] = len(binary)

            for i in range(self.fixed_start_pop['num_pop']):
                self.fixed_start_pop['binary'][i] = binary[i]

            self.fixed_start_pop['decoded'] = self.decode_binary_pop(self.fixed_start_pop['binary'])
            self.fixed_start_pop['scaled'] = self.scale_population(self.fixed_start_pop['decoded'])


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
