import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EP Diet Optimizer", layout="wide")
st.title("🍱 Evolutionary Programming (EP) Diet Optimizer")
st.write("Objective: Minimize RM Cost while keeping Calories strictly between 1200 - 3000.")

# --- 1. DATA PREP ---
@st.cache_data
def load_data():
    df = pd.read_csv("Food_and_Nutrition__.csv")
    df.columns = df.columns.str.strip()
    # Create price in RM
    np.random.seed(42)
    df['Price'] = (df['Calories'] * 0.005 + 2.50).round(2)
    return df

data = load_data()

# --- 2. EP LOGIC ---
def calculate_fitness(plan, target_c):
    t_cal = sum(m['Calories'] for m in plan)
    t_price = sum(m['Price'] for m in plan)
    
    # --- THE HARD LIMIT ---
    # If calories are over 3000, we give it a massive penalty so it LOSES.
    penalty = 0
    if t_cal > 3000 or t_cal < 1200:
        penalty = 1000000 
    
    # Fitness = Price + Penalty + Distance from Target
    # In EP, we want the LOWEST score.
    return t_price + penalty + abs(t_cal - target_c)

def run_ep_optimizer(target_c):
    pools = [
        data[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
        data[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
        data[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
        data[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    ]

    # 1. Initialize Population (Random individuals)
    pop_size = 50
    population = [[p.sample(1).iloc[0] for p in pools] for _ in range(pop_size)]
    
    history = []

    for gen in range(100):
        # 2. Evaluation
        scores = [calculate_fitness(ind, target_c) for ind in population]
        history.append(min(scores) if min(scores) < 1000000 else 50)

        # 3. Selection (Keep the best half)
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices[:pop_size//2]]

        # 4. Mutation (EP only uses mutation, no crossover!)
        # We take the survivors and create "offspring" by slightly changing them
        offspring = []
        for parent in population:
            child = [m.copy() for m in parent]
            # Mutate: Randomly pick 1 meal and replace it
            idx_to_change = np.random.randint(0, 4)
            child[idx_to_change] = pools[idx_to_change].sample(1).iloc[0]
            offspring.append(child)
        
        population.extend(offspring)

    # Return the absolute winner
    winner_scores = [calculate_fitness(ind, target_c) for ind in population]
    return population[np.argmin(winner_scores)], history

# --- 3. UI ---
st.sidebar.header("Nutrition Targets")
user_target = st.sidebar.slider("Calories Target", 1200, 3000, 2000)

if st.button("🚀 Run EP Optimization"):
    winner, history = run_ep_optimizer(user_target)
    
    # Calculate final numbers
    f_cal = sum(m['Calories'] for m in winner)
    f_price = sum(m['Price'] for m in winner)
    f_prot = sum(m['Protein'] for m in winner)

    st.divider()
    # Display results
    c1, c2, c3 = st.columns(3)
    c1.metric("MINIMIZED COST", f"RM {f_price:.2f}")
    c2.metric("TOTAL CALORIES", f"{f_cal} kcal")
    c3.metric("TOTAL PROTEIN", f"{f_prot}g")

    if f_cal > 3000 or f_cal < 1200:
        st.error("Algorithm stuck! Click Run again to refresh the population.")
    else:
        st.success("Success! EP found a valid plan within RM budget.")

    st.subheader("📋 Recommended Plan")
    labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
    col_name = data.columns[0]
    for i in range(4):
        st.write(f"**{labels[i]}:** {winner[i][col_name]} ({winner[i]['Calories']} kcal) - RM {winner[i]['Price']}")

    st.line_chart(history)
