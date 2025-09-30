import random,bisect

# 定义适应度函数
def fitness_function(x):
    return 20*x-x*x

# 初始化种群
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, 15) , random.randint(0, 99)]
        population.append(chromosome)
    return population

# 选择操作 - 轮盘选择
def tournament_selection(population, probability):
    selected = []
    limit=probability[-1]
    for _ in range(2):
        selected.append(population[bisect.bisect_left(probability,random.uniform(0,limit))])
    return selected

# 交叉操作 - 局部交叉
def crossover(parent1, parent2):
    crossover_segment1 , crossover_segment2 = (1<<random.randint(0, 4))-1 , (1<<random.randint(0, 6))-1
    integer_segment = (parent1[0]&crossover_segment1) ^ (parent2[0]&crossover_segment1)
    decimal_segment = (parent1[1]&crossover_segment2) ^ (parent2[1]&crossover_segment2)
    child1 = [parent1[0] ^ integer_segment , (parent1[1] ^ decimal_segment)%100]
    child2 = [parent2[0] ^ integer_segment , (parent2[1] ^ decimal_segment)%100]
    return child1, child2

# 变异操作 - 单点翻转
def mutate(chromosome, mutation_rate):
    for i in range(4):
        if random.random() < mutation_rate:
            chromosome[0] ^= 1<<i
    for i in range(7):
        if random.random() < mutation_rate:
            chromosome[1] ^= 1<<i
    chromosome[1] %= 100
    return chromosome

# 主遗传算法循循环
def genetic_algorithm(population_size, generations, mutation_rate):
    
    population = initialize_population(population_size)
    n = len(population)
    best_fitness , best_chromosome = 0 , 0
    probability = [0]*n

    for gene in range(generations):
        new_population = []

        # 重新评估适应度
        fitness_values = [fitness_function(chromosome[0] + chromosome[1]/100) for chromosome in population]

        # 对比选择当前最优解
        for i in range(n):
            if best_fitness < fitness_values[i]:
                best_fitness = fitness_values[i]
                best_chromosome = population[i][0] + population[i][1]/100

        # 生成概率轮盘
        probability[0] = fitness_values[0]
        for i in range(1,n):
            probability[i] = probability[i-1] + fitness_values[i]

        # 生成下一代
        while len(new_population) < population_size:
            parents = tournament_selection(population, fitness_values)
            child1, child2 = crossover(parents[0], parents[1])
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population

    return best_chromosome
# 主函数
if __name__ == '__main__':
    population_size = 20
    generations = 100
    mutation_rate = 0.05
    best_solution = genetic_algorithm(population_size, generations, mutation_rate)
    print("Best solution found: x={} . The maximum value is y={:.2f}".format(best_solution , fitness_function(best_solution)))
