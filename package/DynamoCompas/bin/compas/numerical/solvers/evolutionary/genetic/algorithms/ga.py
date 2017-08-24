import re
import random
import json
import copy

__author__     = ['Tomas Mendez Echenagucia <mtomas@ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'mtomas@ethz.ch'


class GA:
    """This class contains a binary coded, single objective genetic algorithm. The main function
    is ``GA.ga``, calling this function starts the GA optimization. It also contains all of the
    required genetic operators.
    """

    def __init__(self):
        """ Initializes the GA object.

        Parameters
        ----------
        best_fit: float
            The fitness value of the best performing solution for the current generation.
        best_individual_index: int
            The index of the best performing individual for the current generation.
        boundaries: dict
            This dictionary contains all the max and min bounds for each optimization variable.
            ``GA.boundaries[index] = [min,max]``.
        current_pop: dict
            This dictionary contains the coded, decoded and scaled population of the current
            generation, as well as their fitness values.
        elite_pop: dict
            This dictionary contains the coded, decoded and scaled data for the elite
            population of the current generation, as well as their fitness values.
        end_gen: int
            The maximum number of generations the GA is allowed to run.
        fit_function: function
            The fitness function.
        fit_name: str
            The name of the python file containing the fitness function (without extension).
        fit_type: str
            String that indicates if the fitness function is to be minimized or maximized.
            "min" for minimization and "max" for maximization.
        input_path: str
            Path to the fitness function file.
        kwargs : dict
            This dictionary will be passed as a keyword argument to all fitness functions.
            It can be used to pass required data, objects, that are not related to the
            optimmization variables but is required to run the fitness function.
        max_bin_digit: int
            The maximum number of binary digits that are used to code a variable values.
            The number of binary digits assigned to code a variable determine the number
            of discrete steps inside the variable bounds. For example, an 8 digit binary
             number will produce 256 steps.
        min_fit: float
            An end condition related to fitness value. If it is set, the GA will stop
            when any individual achieves a fitness equal or better that ``GA.min_fit``. If
            it is not set, the GA will end after ``GA.num_gen`` generations.
        min_fit_flag: bool
            Flag the GA uses to determine if the ``GA.min_fit`` value has been achieved.
        mutation_probability: float
            Determines the probability that the mutation operator will mutate each gene.
            For each gene a random number ``x`` between 0 and 1 is generated, if ``x``
            is higher than ``GA.mutation_probability`` it will be mutated.
        num_bin_dig: list
            List of the number of binary digits for each variable. The number of binary
            digits assigned to code a variable determine the number of discrete steps
            inside the variable bounds. For example, an 8 digit binary number will
            produce 256 steps.
        num_elite: int
            The number of top performing individuals in the population that are not subject
            to genetic operators, but are simply copied to the next generation.
        num_gen: int
            The number of genertions the GA will run.
        num_pop: int
            The number of individuals per generation.
        num_var: int
            The number of variables in the optimization problem.
        output_path: str
            The path to which the GA outputs result files.
        start_from_gen: int
            The generation from which the GA will restart. If this number is given, the GA
            will look for generation output files in the ``GA.input_path`` and if found,
            the GA will resume optimization from the ``GA.start_from_gen`` generation.
        total_bin_dig: int
            The total number of binary digits. It is the sum of the ``GA.num_bin_dig`` of
            all variables.
        """

        self.kwargs = {}
        self.best_fit = None
        self.best_individual_index = None
        self.boundaries   = {}
        self.current_pop  = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_value': {}}
        self.elite_pop   = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_value': {}}
        self.end_gen = None
        self.fit_function = None
        self.fit_name = ''
        self.fit_type = None
        self.input_path = None
        self.max_bin_dig = []
        self.min_fit = None
        self.min_fit_flag = False
        self.mutation_probability = 0
        self.num_bin_dig = 0
        self.num_elite = 0
        self.num_gen = 0
        self.num_gen_init_pop = 1
        self.num_pop = 0
        self.num_pop_init = None
        self.num_pop_temp = None
        self.num_var = 0
        self.output_path = []
        self.start_from_gen = False
        self.total_bin_dig = 0
        self.check_diversity = False

    def ga(self):
        """ This is the main optimization function, this function permorms the GA optimization,
        performing all genetic operators.
        """
        self.write_ga_json_file()

        if self.num_pop_init:
            self.num_pop_temp = copy.deepcopy(self.num_pop)
            self.num_pop = self.num_pop_init

        if self.start_from_gen:
            self.current_pop = self.get_pop_from_pop_file(self.start_from_gen)
            start_gen_number = self.start_from_gen + 1
        else:
            self.current_pop['binary'] = self.generate_random_bin_pop()
            start_gen_number = 0

        for generation in range(start_gen_number, self.num_gen):

            self.current_pop['decoded'] = self.decode_binary_pop(self.current_pop['binary'])
            self.current_pop['scaled']  = self.scale_population(self.current_pop['decoded'])

            if generation == 0:
                num = self.num_pop
            else:
                num = self.num_pop - self.num_elite
            for i in range(num):
                self.current_pop['fit_value'][i] = self.fit_function(self.current_pop['scaled'][i], **self.kwargs)

            if self.num_pop_init and generation >= self.num_gen_init_pop:
                self.num_pop = self.num_pop_temp
                self.current_pop = self.select_elite_pop(self.current_pop, num_elite=self.num_pop)

            self.write_out_file(generation)

            if self.min_fit:
                self.update_min_fit_flag()
            else:
                self.get_best_fit()

            print ('generation ', generation, ' opt fit ', self.best_fit, 'min fit', self.min_fit)
            if self.check_diversity:
                print ('num repeated individuals', self.check_pop_diversity())
            if generation < self.num_gen - 1 and self.min_fit_flag is False:
                self.elite_pop = self.select_elite_pop(self.current_pop)
                self.tournament_selection()  # n-e
                self.create_mating_pool()  # n-e
                self.simple_crossover()  # n-e
                self.random_mutation()  # n-e
                self.add_elite_to_current()  # n
            else:
                self.end_gen = generation
                self.get_best_individual_index()
                self.write_ga_json_file()
                print ('GA ended at generation ', self.end_gen, ' with an optimal fitness of ', self.best_fit)
                break

    def check_pop_diversity(self):
        seen = []
        all_ = []
        for key in self.current_pop['scaled']:
            ind_ = self.current_pop['scaled'][key]
            ind = [ind_[k] for k in ind_]
            if ind not in seen:
                seen.append(ind)
            all_.append(ind)
        return len(all_) - len(seen)

    def decode_binary_pop(self, bin_pop):
        """Decodes the binary population, from binary to unscaled variable values

        Parameters
        ----------
        bin_pop: dict
            The binary population dictionary.

        Returns
        -------
        decoded_pop:
            The decoded population dictionary.
        """
        decoded_pop = {}
        for j in range(self.num_pop):
            decoded_pop[j] = {}
            for i in range(self.num_var):
                value = 0
                chrom = bin_pop[j][i]
                for u, gene in enumerate(chrom):
                    if gene == 1:
                        value = value + 2**u
                decoded_pop[j][i] = value
        return decoded_pop

    def generate_random_bin_pop(self):
        """ Generates random binary population of ``GA.num_pop`` size.

        Returns
        -------
        random_bin_pop: dict
            A dictionary containing a random binary population.
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

    def scale_population(self, decoded_pop):

        """Scales the decoded population, variable values are scaled according to each
        of their bounds contained in ``GA.boundaries``.

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

        for j in range(self.num_pop):
            scaled_pop[j] = {}
            for i in range(self.num_var):
                scaled_pop[j][i] = 1.0 / max_bin[i] * decoded_pop[j][i]

        for j in range(self.num_pop):
            for i in range(self.num_var):
                bound = self.boundaries[i]
                scaled_pop[j][i] = bound[0] + (bound[1] - bound[0]) * scaled_pop[j][i]

        return scaled_pop

    def tournament_selection(self):
        """Performs the tournament selection operator on the current population.
        """
        pop_a = []
        pop_b = []
        indices = range(self.num_pop)
        for i in range((self.num_pop - self.num_elite)):
            u, v = random.sample(indices, 2)
            pop_a.append(u)
            pop_b.append(v)
        # pop_a = random.sample(indices,self.num_pop-self.num_elite)
        # pop_b = random.sample(indices,self.num_pop-self.num_elite)
        self.mp_indices = []
        for i in range(self.num_pop - self.num_elite):
            fit_a = self.current_pop['fit_value'][pop_a[i]]
            fit_b = self.current_pop['fit_value'][pop_b[i]]
            if self.fit_type == 'min':
                if fit_a < fit_b:
                    self.mp_indices.append(pop_a[i])
                else:
                    self.mp_indices.append(pop_b[i])
            elif self.fit_type == 'max':
                if fit_a > fit_b:
                    self.mp_indices.append(pop_a[i])
                else:
                    self.mp_indices.append(pop_b[i])

    def select_elite_pop(self, pop, num_elite=None):
        """Saves the elite population in the elite population dictionary

        Parameters
        ----------
        pop: dict
            A population dictionary

        Returns
        -------
        elite_pop: dict
            The elite population dictionary.
        """
        elite_pop   = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_value': {}}

        # fit_list = pop['fit_value'].values()
        fit_list = [self.current_pop['fit_value'][i] for i in range(len(self.current_pop['fit_value']))]

        if self.fit_type == 'min':
            index_list = self.get_sorting_indices(fit_list)
        elif self.fit_type == 'max':
            index_list = self.get_sorting_indices(fit_list, reverse=True)
        else:
            raise ValueError('User selected fit_type is wrong. Use "min" or "max" only')
        if not num_elite:
            num_elite = self.num_elite
        for i in range(num_elite):
            elite_pop['binary'][i] = pop['binary'][index_list[i]]
            elite_pop['decoded'][i] = pop['decoded'][index_list[i]]
            elite_pop['scaled'][i] = pop['scaled'][index_list[i]]
            elite_pop['fit_value'][i] = pop['fit_value'][index_list[i]]

        return elite_pop

    def get_sorting_indices(self, l, reverse=False):
        """Reurns the indices that would sort a list of floats. If floats are
        repeated in the list, only one instance is considered. The index of
        repeaded floats are included in the end of the index list.

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
        l_ = []
        if reverse:
            x = str('-inf')
        else:
            x = str('inf')
        for i in l:
            if i in l_:
                l_.append(x)
            else:
                l_.append(i)
        sorting_index = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(l_))]
        if reverse is True:
            sorting_index = list(reversed(sorting_index))
        return sorting_index

    def create_mating_pool(self):
        """Creates two lists of cromosomes to be used by the crossover operator.
        """
        self.mating_pool_a = []
        self.mating_pool_b = []
        for i in range((self.num_pop - self.num_elite) / 2):
            chrom_a = []
            chrom_b = []
            for j in range(self.num_var):
                chrom_a += self.current_pop['binary'][self.mp_indices[i]][j]
                chrom_b += self.current_pop['binary'][self.mp_indices[i + ((self.num_pop - self.num_elite) / 2)]][j]
            self.mating_pool_a.append(chrom_a)
            self.mating_pool_b.append(chrom_b)

    def simple_crossover(self):
        """Performs the simple crossover operator. Individuals in ``GA.mating_pool_a`` are
        combined with individuals in ``GA.mating_pool_b`` using a single, randomly selected
        crossover point.
        """

        self.current_pop  = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_value': {}}

        for j in range((self.num_pop - self.num_elite) / 2):
            cross = random.randint(1, self.total_bin_dig - 1)
            a = self.mating_pool_a[j]
            b = self.mating_pool_b[j]
            c = a[:cross] + b[cross:]
            d = b[:cross] + c[cross:]

            self.current_pop['binary'][j] = {}
            self.current_pop['binary'][j + ((self.num_pop - self.num_elite) / 2)] = {}
            for i in range(self.num_var):
                variable_a = c[:self.num_bin_dig[i]]
                del c[:self.num_bin_dig[i]]
                variable_b = d[:self.num_bin_dig[i]]
                del d[:self.num_bin_dig[i]]
                self.current_pop['binary'][j][i] = variable_a
                self.current_pop['binary'][j + ((self.num_pop - self.num_elite) / 2)][i] = variable_b

    def random_mutation(self):
        """This mutation operator replaces a gene from 0 to 1 or viceversa
        with a probability of ``GA.mutation_probability``.
        """
        for i in range(self.num_pop - self.num_elite):
            for j in range(self.num_var):
                for u in range(self.num_bin_dig[j]):
                    random_value = random.random()
                    if random_value < (self.mutation_probability):
                        if self.current_pop['binary'][i][j][u] == 0:
                            self.current_pop['binary'][i][j][u] = 1
                        else:
                            self.current_pop['binary'][i][j][u] = 0

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
        for i in range(self.num_pop):
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

        for i in range(self.num_pop):
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

    def write_out_file(self, generation):
        """Writes the population data for a given generation.

        Parameters
        ----------
        generation: int
            The generation number.
        """
        filename  = 'generation_' + "%05d" % generation + '_population' + ".pop"
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
                pf_file.write(str(self.current_pop['scaled'][i][f]))
                pf_file.write(',')
            pf_file.write('\n')
        pf_file.write('\n')

        pf_file.write('Population fitness value \n')
        for i in range(self.num_pop):
            pf_file.write(str(i) + ',')
            pf_file.write(str(self.current_pop['fit_value'][i]))
            pf_file.write('\n')
        pf_file.write('\n')
        pf_file.write('\n')
        pf_file.close()

    def add_elite_to_current(self):
        """Adds the elite population to the current population dictionary.
        """
        for i in range(self.num_elite):
            self.current_pop['binary'][self.num_pop - self.num_elite + i] = self.elite_pop['binary'][i]
            self.current_pop['decoded'][self.num_pop - self.num_elite + i] = self.elite_pop['decoded'][i]
            self.current_pop['scaled'][self.num_pop - self.num_elite + i] = self.elite_pop['scaled'][i]
            self.current_pop['fit_value'][self.num_pop - self.num_elite + i] = self.elite_pop['fit_value'][i]

    def make_ga_input_data(self):
        """Returns a dictionary containing the most relavant genetic data present in the instance
        of ``GA``. This is the data required to restart a genetic optimization process or to
        launch a visualization using ``compas_ga.visualization.ga_visualization``.

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
                'fit_name': self.fit_name,
                'fit_type': self.fit_type,
                'start_from_gen': self.start_from_gen,
                'max_bin_dig': self.max_bin_dig,
                'total_bin_dig': self.total_bin_dig,
                'output_path': self.output_path,
                'num_elite': self.num_elite,
                'min_fit': self.min_fit,
                'end_gen': self.end_gen,
                'best_individual_index': self.best_individual_index
                }
        return data

    def write_ga_json_file(self):
        """Writes a JSON file containing the most relevant data for GA optimization and
        visualization using ``compas_ga.visualization.ga_visualization``.
        """
        data = self.make_ga_input_data()
        filename = self.fit_name + '.json'
        with open(self.output_path + filename, 'wb+') as fh:
            json.dump(data, fh)

    def update_min_fit_flag(self):
        """Checks if the minimum desired fitness value has been achieved during optimization
        and saves the result in ``GA.min_fit_flag``.
        """
        values = self.current_pop['fit_value'].values()
        if self.fit_type == 'min':
            self.best_fit = min(values)
            if self.best_fit <= self.min_fit:
                self.min_fit_flag = True
        elif self.fit_type == 'max':
            self.best_fit = max(values)
            if self.best_fit >= self.min_fit:
                self.min_fit_flag = True

    def get_best_fit(self):
        """Saves the best fitness value in ``GA.best_fit``
        """
        if self.fit_type == 'min':
            self.best_fit = min(self.current_pop['fit_value'].values())
        elif self.fit_type == 'max':
            self.best_fit = max(self.current_pop['fit_value'].values())

    def get_pop_from_pop_file(self, gen):
        """Reads the population file corresponding to the ``gen`` generation and returns
        the saved population data. The population file must be in ``GA.input_path``.

        Parameters
        ----------
        gen: int
            The generation number.

        Returns
        -------
        file_pop: dict
            The population dictionary contained in the file.
        """
        file_pop  = {'binary': {}, 'decoded': {}, 'scaled': {}, 'fit_value': {},
                     'pf': {}}
        filename  = 'generation_' + "%05d" % gen + '_population' + ".pop"
        filename = self.input_path + filename
        pf_file = open(filename, 'r')
        lines = pf_file.readlines()
        pf_file.close()

        for i in range(self.num_pop):
            file_pop['scaled'][i] = {}
            file_pop['fit_value'][i] = {}
            line_scaled = lines[i + 7]
            line_fit = lines[i + 9 + self.num_pop]
            string_scaled = re.split(',', line_scaled)
            string_fit = re.split(',', line_fit)
            string_fit = string_fit[1]
            del string_scaled[-1]
            del string_scaled[0]
            scaled = [float(j) for j in string_scaled]
            fit_value = float(string_fit)
            for j in range(len(scaled)):
                file_pop['scaled'][i][j] = scaled[j]
            file_pop['fit_value'][i] = fit_value

        file_pop['decoded'] = self.unscale_pop(file_pop['scaled'])
        file_pop['binary'] = self.code_decoded(file_pop['decoded'])
        return file_pop

    def get_best_individual_index(self):
        """Saves the index of the best performing individual of the current population
         in ``GA.best_individual_index``.
        """
        # fit_values = self.current_pop['fit_value'].values()
        fit_values = [self.current_pop['fit_value'][i] for i in range(len(self.current_pop['fit_value']))]
        print('fit_values', fit_values)
        if self.fit_type == 'min':
            indices = self.get_sorting_indices(fit_values)
        elif self.fit_type == 'max':
            indices = self.get_sorting_indices(fit_values, reverse=True)
        print('fitness sorted individual indices', indices)
        self.best_individual_index = indices[0]
        print('self.best_individual_index', self.best_individual_index)


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
