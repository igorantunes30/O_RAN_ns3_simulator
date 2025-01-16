import numpy as np
import random
import matplotlib.pyplot as plt

def calculate_energy_total(A_t, B_t, S_t, 
                           L_epm, L_cpm, 
                           P_epm, P_prime_epm, C_epm, 
                           P_cpm, P_prime_cpm, C_cpm, 
                           V_du, V_cu, alpha, beta, T):
    """
    Função para calcular o consumo total de energia no intervalo de tempo t.

    Parâmetros:
    A_t (numpy.array): Matriz de associação DU-EPM.
    B_t (numpy.array): Matriz de associação CU-CPM.
    S_t (numpy.array): Matriz de divisões funcionais CU-DU.
    L_epm (numpy.array): Carga computacional nos EPMs.
    L_cpm (numpy.array): Carga computacional nos CPMs.
    P_epm (float): Consumo estático de energia dos EPMs.
    P_prime_epm (float): Consumo dinâmico de energia dos EPMs.
    C_epm (float): Capacidade computacional dos EPMs.
    P_cpm (float): Consumo estático de energia dos CPMs.
    P_prime_cpm (float): Consumo dinâmico de energia dos CPMs.
    C_cpm (float): Capacidade computacional dos CPMs.
    V_du (numpy.array): Volume de dados migrados pelas DUs.
    V_cu (numpy.array): Volume de dados migrados pelas CUs.
    alpha (float): Coeficiente de migração para tráfego.
    beta (float): Coeficiente fixo de energia de migração.
    T (float): Intervalo de tempo para cálculo.

    Retorna:
    float: Energia total calculada.
    """
    A_t = np.array(A_t)
    if A_t.ndim == 1:
        A_t = A_t[:, np.newaxis]

    # Energia relacionada ao processamento nos EPMs
    E_processing_epm = np.sum(
        (L_epm > 0) * P_epm + P_prime_epm * (L_epm / C_epm)
    ) * T

    # Energia relacionada ao processamento nos CPMs
    E_processing_cpm = np.sum(
        (L_cpm > 0) * P_cpm + P_prime_cpm * (L_cpm / C_cpm)
    ) * T

    # Energia relacionada à migração nos EPMs
    if A_t.shape[0] > 1:
        migration_epm = np.sum((1 - A_t[:-1]) * A_t[1:] * (alpha * V_du + beta))
    else:
        migration_epm = 0

    # Energia relacionada à migração nos CPMs
    if B_t.shape[0] > 1:
        migration_cpm = np.sum((1 - B_t[:-1]) * B_t[1:] * (alpha * V_cu[:B_t.shape[1]] + beta))
    else:
        migration_cpm = 0

    E_migration_epm = migration_epm
    E_migration_cpm = migration_cpm

    # Energia total
    E_total = (E_processing_epm + E_processing_cpm +
               E_migration_epm + E_migration_cpm)

    return E_total

def flower_pollination_algorithm(max_generations, population_size, bounds, 
                                 L_epm, L_cpm, P_epm, P_prime_epm, C_epm, 
                                 P_cpm, P_prime_cpm, C_cpm, V_du, V_cu, alpha, beta, T):
    """
    Implementação do algoritmo de polinização por flores (FPA) para otimização.

    Parâmetros:
    max_generations (int): Número máximo de gerações.
    population_size (int): Tamanho da população.
    bounds (list of tuple): Limites das variáveis de decisão.

    Retorna:
    tuple: Melhor solução encontrada e seu valor objetivo.
    """
    # Inicializar população
    population = [np.array([random.uniform(b[0], b[1]) for b in bounds]) for _ in range(population_size)]

    # Inicializar matrizes B_t e S_t como exemplos para uso no cálculo
    B_t = np.zeros((len(population[0]), 1))  # Ajustando para múltiplas dimensões
    S_t = np.ones((len(population[0]), 1))   # Ajustando para múltiplas dimensões

    fitness = [calculate_energy_total(A_t, B_t, S_t, 
                                      L_epm, L_cpm, 
                                      P_epm, P_prime_epm, C_epm, 
                                      P_cpm, P_prime_cpm, C_cpm, 
                                      V_du, V_cu, alpha, beta, T) for A_t in population]

    best_solution = population[np.argmin(fitness)]
    best_fitness = min(fitness)

    # Armazenar valores de fitness para o gráfico de convergência
    convergence = [best_fitness]

    for generation in range(max_generations):
        for i in range(population_size):
            if random.random() < 0.8:  # Polinização global
                L = random.gauss(0, 1)
                step = L * (population[i] - best_solution)
                population[i] = population[i] + step
            else:  # Polinização local
                epsilon = random.random()
                j, k = random.sample(range(population_size), 2)
                step = epsilon * (population[j] - population[k])
                population[i] = population[i] + step

            # Respeitar os limites das variáveis
            population[i] = np.clip(population[i], [b[0] for b in bounds], [b[1] for b in bounds])

        fitness = [calculate_energy_total(A_t, B_t, S_t, 
                                          L_epm, L_cpm, 
                                          P_epm, P_prime_epm, C_epm, 
                                          P_cpm, P_prime_cpm, C_cpm, 
                                          V_du, V_cu, alpha, beta, T) for A_t in population]

        current_best = min(fitness)
        if current_best < best_fitness:
            best_fitness = current_best
            best_solution = population[np.argmin(fitness)]

        convergence.append(best_fitness)
        print(f"Geração {generation + 1}: Melhor fitness = {best_fitness}")

    # Plotar gráfico de convergência
    plt.figure(figsize=(10, 6))
    plt.plot(convergence, label="Fitness")
    plt.title("Convergência do Algoritmo de Polinização por Flores")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.legend()
    plt.grid()
    plt.show()

    return best_solution, best_fitness

# Limites das variáveis (exemplo para A_t)
bounds = [(0, 1) for _ in range(4)]  # Ajuste conforme o problema

# Parâmetros de exemplo
L_epm = np.array([50, 60])  # Carga computacional nos EPMs
L_cpm = np.array([70, 80])  # Carga computacional nos CPMs
P_epm = 200  # Consumo estático dos EPMs
P_prime_epm = 50  # Consumo dinâmico dos EPMs
C_epm = 100  # Capacidade computacional dos EPMs
P_cpm = 300  # Consumo estático dos CPMs
P_prime_cpm = 60  # Consumo dinâmico dos CPMs
C_cpm = 120  # Capacidade computacional dos CPMs
V_du = np.array([5, 10])  # Volume de dados migrados pelas DUs
V_cu = np.array([15, 20])  # Volume de dados migrados pelas CUs
alpha = 0.5  # Coeficiente de migração
beta = 10  # Coeficiente fixo de migração
T = 1  # Intervalo de tempo

# Executar o algoritmo
best_solution, best_fitness = flower_pollination_algorithm(100, 20, bounds, 
                                                           L_epm, L_cpm, P_epm, P_prime_epm, C_epm, 
                                                           P_cpm, P_prime_cpm, C_cpm, V_du, V_cu, alpha, beta, T)

print("Melhor solução encontrada:", best_solution)
print("Melhor valor objetivo:", best_fitness)
