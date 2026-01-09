import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer", layout="wide")
st.title("🥗 Diet Meal Planning Optimisation using Evolutionary Programming")

# --- DATA LOADING ---
if os.path.exists("Food_and_Nutrition__.csv"):
    data = pd.read_csv("Food_and_Nutrition__.csv")
    
    # Ensure column names match your CSV exactly
    # Creating a synthetic price based on calories for the optimization objective
    np.random.seed(42)
    data['Price'] = (data['Calories'] * 0.005 + np.random.uniform(1, 3, size=len(data))).round(2)

    # --- STEP 1: USER INPUTS ---
    st.subheader("Set Daily Nutrition Targets")
    col_in1, col_in2, col_in3 = st.columns(3)
    cal_target = col_in1.number_input("Calories", value=2000, step=100)
    protein_target = col_in2.number_input("Protein (g)", value=75, step=5)
    fat_target = col_in3.number_input("Fat (g)", value=70, step=5)

    # --- STEP 2: EVOLUTIONARY PROGRAMMING PARAMETERS ---
    POP_SIZE = 50
    GENERATIONS = 100
    MUTATION_RATE = 0.2

    # Pools for combination (Breakfast, Lunch, Dinner, Snack)
    b_pool = data[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    l_pool = data[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    d_pool = data[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    s_pool = data[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()

    # --- STEP 3: FITNESS FUNCTION ---
    def fitness(meals):
        t_cal = sum(m['Calories'] for m in meals)
        t_prot = sum(m['Protein'] for m in meals)
        t_fat = sum(m['Fat'] for m in meals)
        t_price = sum(m['Price'] for m in meals)

        # Multi-Objective Penalty: Difference from targets
        penalty = abs(t_cal - cal_target) * 2
        penalty += abs(t_prot - protein_target) * 5
        penalty += abs(t_fat - fat_target) * 5
        
        # Goal: Minimize (Price + Penalties)
        return t_price + penalty, t_cal, t_prot, t_fat, t_price

    # --- STEP 4: EVOLUTIONARY LOOP ---
    def run_ep():
        # Initialize population
        pop = []
        for _ in range(POP_SIZE):
            ind = [b_pool.sample(n=1).iloc[0], l_pool.sample(n=1).iloc[0], 
                   d_pool.sample(n=1).iloc[0], s_pool.sample(n=1).iloc[0]]
            pop.append(ind)
        
        history = []
        for gen in range(GENERATIONS):
            # Evaluate fitness
            scores = [fitness(ind)[0] for ind in pop]
            history.append(min(scores))
            
            # Selection (Keep best half)
            pop = [pop[i] for i in np.argsort(scores)[:POP_SIZE//2]]
            
            # Mutation (Duplicate and mutate)
            children = []
            for parent in pop:
                child = list(parent)
                if np.random.rand() < MUTATION_RATE:
                    idx = np.random.randint(0, 4)
                    pools = [b_pool, l_pool, d_pool, s_pool]
                    child[idx] = pools[idx].sample(n=1).iloc[0]
                children.append(child)
            pop.extend(children)
            
        # Return best
        final_scores = [fitness(ind)[0] for ind in pop]
        return pop[np.argmin(final_scores)], history

    # --- STEP 5: OUTPUT ---
    if st.button("Run Evolutionary Programming"):
        best_plan, fitness_history = run_ep()
        f_score, f_cal, f_prot, f_fat, f_price = fitness(best_plan)

        # Dashboard Metrics
        st.divider()
        st.subheader("Optimization Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Price", f"${f_price:.2f}")
        m2.metric("Total Calories", f"{f_cal} kcal")
        m3.metric("Total Protein", f"{f_prot}g")
        m4.metric("Total Fat", f"{f_fat}g")

        # Meal Plan Table
        st.write("### Recommended Combination")
        meal_names = ["Breakfast", "Lunch", "Dinner", "Snack"]
        suggestion_cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        
        for i in range(4):
            st.info(f"**{meal_names[i]}:** {best_plan[i][suggestion_cols[i]]} | Cost: ${best_plan[i]['Price']}")

        # Convergence Plot
        st.subheader("Convergence Analysis")
        fig, ax = plt.subplots()
        ax.plot(fitness_history, color='blue', label='Best Fitness Score')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (Penalty + Price)")
        ax.legend()
        st.pyplot(fig)

else:
    st.error("CSV File 'Food_and_Nutrition__.csv' not found!")
