import numpy as np
from sklearn.naive_bayes import GaussianNB

def cal_pop_fitness(pop, X_train, y_train, X_test, y_test):
  accuracies = np.zeros(pop.shape[0])

  for i, current_solution in enumerate(pop):
    classifier = GaussianNB()
    classifier.fit(X_train[:, current_solution == 1], y_train)
    accuracy = classifier.score(X_test[:, current_solution == 1], y_test)
    accuracies[i] = accuracy

  return accuracies

def select_mating_pool(pop, fitness, num_parents):
  # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
  parents = np.empty((num_parents, pop.shape[1]))
  for parent_num in range(num_parents):
    max_fitness_idx = np.where(fitness == np.max(fitness))
    max_fitness_idx = max_fitness_idx[0][0]
    parents[parent_num, :] = pop[max_fitness_idx, :]
    fitness[max_fitness_idx] = -999999
  return parents

def crossover(parents, offspring_size):
  offspring = np.empty(offspring_size)
  # The point at which crossover takes place between 2 parents. Usually, it is at the center.
  crossover_point = np.uint8(offspring_size[1]/2)

  for k in range(offspring_size[0]):
    # Index of the first parent to mate.
    parent1_idx = k % parents.shape[0]
    # Index of the second parent to mate.
    parent2_idx = (k + 1) % parents.shape[0]
    # The new offspring will have its first half of its genes taken from the first parent.
    offspring[k, 0: crossover_point] = parents[parent1_idx, 0: crossover_point]
    # The new offspring will have its second half of its genes taken from the second parent.
    offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
  return offspring

def mutation(offspring_crossover, num_mutations=2):
  mutation_idx = np.random.randint(low=0, high=offspring_crossover.shape[1], size=num_mutations)
  # Mutation changes a single gene in each offspring randomly
  offspring_crossover[:, mutation_idx] = 1 - offspring_crossover[:, mutation_idx]
  return offspring_crossover