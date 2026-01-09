import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer RM", layout="wide")
st.title("🍱 Diet Meal Planning (Strict Constraints)")

# Load CSV
if os.path.exists("Food_and_Nutrition__.csv"):
    data = pd.read_csv("Food_and_Nutrition__.csv")

    # Generate synthetic Price in RM if not exists
    if 'Price' not in data.columns:
        np.random.seed(42)
        data['Price'] = (data['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(data))).round(2)

    # --- USER INPUTS ---
    st.subheader("Set Daily Nutrition Targets")
    col1, col2, col3 = st.columns(3)
    cal_target = col1.number_input("Target Calories", value=2000, step=50)
    protein_target = col2.number_input("Protein (g)", value=75)
    fat_target = col3.number_input("Fat (g)", value=70)

    # Pools
    b_pool = data[['Breakfast Suggestion','Calories','Protein','Fat','Price']].dropna()
    l_pool = data[['Lunch Suggestion','Calories','Protein','Fat','Price']].dropna()
    d_pool = data[['Dinner Suggestion','Calories','Protein','Fat','Price']].dropna()
    s_pool = data[['Snack Suggestion','Calories','Protein','Fat','Price']].dropna()

    # --- FITNESS FUNCTION ---
    def fitness(plan):
        total_cal = sum(m['Calories'] for m in plan)
        total_prot = sum(m['Protein'] for m in plan)
        total_fat = sum(m['Fat'] for m in plan)
        total_price = sum(m['Price'] for m in plan)

        penalty = 0
        # Hard constraints: calories must be reasonable
        if total_cal < 1200 or total_cal > 3000:
            penalty += 1000 * abs(total_cal - cal_target)
        # Soft penalties
        penalty += abs(total_cal - cal_target) * 5
        penalty += abs(total_prot - protein_target) * 5
        penalty += abs(total_fat - fat_target) * 5

        # Objective: minimize price + penalty
        score = total_price + penalty
        return score

    # --- EVOLUTIONARY PROGRAMMING ---
    def ep_optimizer(pop_size=50, generations=50):
        # Initialize population
        population = [
            [
                b_pool.sample(n=1).iloc[0],
                l_pool.sample(n=1).iloc[0],
                d_pool.sample(n=1).iloc[0],
                s_pool.sample(n=1).iloc[0]
            ] for _ in range(pop_size)
        ]

        history = []
        for gen in range(generations):
            # Evaluate fitness
            scores = np.array([fitness(ind) for ind in population])
            history.append(scores.min())

            # Selection: keep top 50%
            idx_sorted = np.argsort(scores)
            population = [population[i] for i in idx_sorted[:pop_size//2]]

            # Mutation: change one random meal in each parent
            children = []
            for parent in population:
                child = list(parent)
                idx = np.random.randint(0, 4)
                pools = [b_pool, l_pool, d_pool, s_pool]
                child[idx] = pools[idx].sample(n=1).iloc[0]
                children.append(child)

            population.extend(children)

        # Return best individual and history
        final_scores = np.array([fitness(ind) for ind in population])
        best_idx = np.argmin(final_scores)
        return population[best_idx], history

    # --- RUN EP ON BUTTON CLICK ---
    if st.button("🚀 Find My Optimized Meal Plan"):
        best_plan, history = ep_optimizer(pop_size=80, generations=100)
        total_cal = sum(m['Calories'] for m in best_plan)
        total_prot = sum(m['Protein'] for m in best_plan)
        total_fat = sum(m['Fat'] for m in best_plan)
        total_price = sum(m['Price'] for m in best_plan)

        st.subheader("Optimized Daily Result (RM)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Daily Cost", f"RM {total_price:.2f}")
        c2.metric("Total Calories", f"{total_cal} kcal")
        c3.metric("Total Protein", f"{total_prot} g")
        c4.metric("Total Fat", f"{total_fat} g")

        st.write("### 📋 Recommended Menu")
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        for i in range(4):
            st.success(f"**{labels[i]}:** {best_plan[i][cols[i]]} "
                       f"({best_plan[i]['Calories']} kcal, {best_plan[i]['Protein']}g protein) - RM {best_plan[i]['Price']:.2f}")

        # --- Convergence Chart ---
        st.subheader("Convergence over Generations")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, color='purple')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (Lower is Better)")
        st.pyplot(fig)

else:
    st.error("Please ensure 'Food_and_Nutrition__.csv' is in the folder.")
