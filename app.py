import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Diet Meal Planning Optimisation using Evolutionary Programming")

# Load dataset
data = pd.read_csv("Food_and_Nutrition__.csv")

st.subheader("Food and Nutrition Dataset")
st.dataframe(data)

# --- EP PARAMETERS ---
POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1

# Nutritional targets
TARGET_CALORIES = 2000
TARGET_PROTEIN = 50
TARGET_FAT = 70

# Fitness function: how well a meal plan meets nutrition while minimizing cost
def fitness(plan):
    total_calories = np.sum(plan * data['Calories'].values)
    total_protein = np.sum(plan * data['Protein'].values)
    total_fat = np.sum(plan * data['Fat'].values)
    total_price = np.sum(plan * data['Price'].values)
    
    # Penalize deviation from targets
    calorie_penalty = abs(TARGET_CALORIES - total_calories)
    protein_penalty = abs(TARGET_PROTEIN - total_protein)
    fat_penalty = abs(TARGET_FAT - total_fat)
    
    return calorie_penalty + protein_penalty + fat_penalty + total_price

# --- EP ALGORITHM ---
def ep_optimizer():
    num_foods = len(data)
    
    # Initialize population with random meal counts (0 or 1)
    pop = np.random.randint(0, 2, size=(POP_SIZE, num_foods))
    fitness_history = []

    for gen in range(GENERATIONS):
        fitness_values = np.array([fitness(ind) for ind in pop])
        fitness_history.append(np.min(fitness_values))
        
        # Selection: top 50% survive
        top_idx = fitness_values.argsort()[:POP_SIZE//2]
        top_pop = pop[top_idx]
        
        # Mutation to refill population
        new_pop = []
        while len(new_pop) < POP_SIZE:
            parent = top_pop[np.random.randint(len(top_pop))].copy()
            # Mutate each gene with MUTATION_RATE
            for i in range(num_foods):
                if np.random.rand() < MUTATION_RATE:
                    parent[i] = 1 - parent[i]  # flip 0 ↔ 1
            new_pop.append(parent)
        
        pop = np.array(new_pop)
    
    # Return best plan
    fitness_values = np.array([fitness(ind) for ind in pop])
    best_idx = np.argmin(fitness_values)
    return pop[best_idx], fitness_history

# --- RUN OPTIMIZER ---
if st.button("Run Evolutionary Programming Optimizer"):
    best_plan, fitness_history = ep_optimizer()
    
    st.subheader("Recommended Meals")
    selected_meals = data[best_plan == 1]
    if len(selected_meals) == 0:
        st.write("No meals selected. Try running again.")
    else:
        for idx, row in selected_meals.iterrows():
            st.write(f"🍽 {row['Food']} - Calories: {row['Calories']}, Protein: {row['Protein']}, Fat: {row['Fat']}, Price: ${row['Price']:.2f}")
        
        total_calories = np.sum(selected_meals['Calories'])
        total_protein = np.sum(selected_meals['Protein'])
        total_fat = np.sum(selected_meals['Fat'])
        total_price = np.sum(selected_meals['Price'])
        
        st.write(f"**Total Calories:** {total_calories}")
        st.write(f"**Total Protein:** {total_protein}")
        st.write(f"**Total Fat:** {total_fat}")
        st.write(f"**Total Cost:** ${total_price:.2f}")
    
    # Plot convergence
    st.subheader("Fitness Convergence Over Generations")
    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Lower is Better)")
    st.pyplot(fig)

