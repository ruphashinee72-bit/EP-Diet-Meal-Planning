import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Diet Meal Planning Optimisation using Evolutionary Programming")

# Load dataset
data = pd.read_csv("Food_and_Nutrition__.csv")

# Add Price column (random reasonable prices for example)
np.random.seed(42)
data['Price'] = np.random.randint(2, 10, size=len(data))  # $2-$9 per meal

st.subheader("Food and Nutrition Dataset with Price")
st.dataframe(data)

# Step 1: User inputs nutrition targets
st.subheader("Set Daily Nutrition Targets")
cal_target = st.number_input("Calories", value=2000, min_value=1000, max_value=5000)
protein_target = st.number_input("Protein (g)", value=75, min_value=10, max_value=300)
fat_target = st.number_input("Fat (g)", value=70, min_value=10, max_value=200)

# Step 2: Evolutionary Programming parameters
POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.2

# Helper function: fitness = how well a meal plan meets targets while minimizing price
def fitness(plan):
    total_cal = np.sum(plan * data['Calories'].values)
    total_protein = np.sum(plan * data['Protein (g)'].values)
    total_fat = np.sum(plan * data['Fat (g)'].values)
    total_price = np.sum(plan * data['Price'].values)

    # Penalty if constraints not met
    penalty = 0
    if total_cal < cal_target:
        penalty += (cal_target - total_cal) * 2
    if total_protein < protein_target:
        penalty += (protein_target - total_protein) * 5
    if total_fat < fat_target:
        penalty += (fat_target - total_fat) * 3

    return total_price + penalty  # lower fitness is better

# Initialize population (binary: include meal or not)
def init_population():
    return np.random.randint(0, 2, size=(POP_SIZE, len(data)))

# Evolutionary Programming main loop
def ep_optimizer():
    pop = init_population()
    best_fitness_list = []

    for gen in range(GENERATIONS):
        fitness_values = np.array([fitness(ind) for ind in pop])
        best_fitness_list.append(fitness_values.min())

        # Selection: keep best half
        idx = np.argsort(fitness_values)
        pop = pop[idx[:POP_SIZE//2]]

        # Mutation: duplicate + mutate
        new_pop = []
        for ind in pop:
            child = ind.copy()
            for i in range(len(child)):
                if np.random.rand() < MUTATION_RATE:
                    child[i] = 1 - child[i]  # flip 0/1
            new_pop.append(child)
        pop = np.vstack([pop, new_pop])  # new generation

    # Return best individual
    final_fitness = np.array([fitness(ind) for ind in pop])
    best_idx = final_fitness.argmin()
    return pop[best_idx], best_fitness_list

# Step 3: Run optimizer
if st.button("Run Evolutionary Programming"):
    best_plan, fitness_history = ep_optimizer()

    # Show recommended meals
    st.subheader("Recommended Daily Meal Plan")
    recommended = data[best_plan==1].copy()
    st.dataframe(recommended)

    st.write("Total Calories:", recommended['Calories'].sum())
    st.write("Total Protein (g):", recommended['Protein (g)'].sum())
    st.write("Total Fat (g):", recommended['Fat (g)'].sum())
    st.write("Total Price ($):", recommended['Price'].sum())

    # Convergence plot
    st.subheader("Convergence over Generations")
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Lower is Better)")
    st.pyplot(plt)
