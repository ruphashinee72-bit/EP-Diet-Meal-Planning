import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer RM", layout="wide")
st.title("🍱 Diet Meal Planning (Evolutionary Programming)")

# --- LOAD DATA ---
if os.path.exists("Food_and_Nutrition__.csv"):
    data = pd.read_csv("Food_and_Nutrition__.csv")
    
    # Ensure Calories, Protein, Fat columns exist
    if 'Calories' not in data.columns or 'Protein' not in data.columns or 'Fat' not in data.columns:
        st.error("CSV must contain 'Calories', 'Protein', and 'Fat' columns.")
    
    # Generate synthetic Price column in RM
    np.random.seed(42)
    data['Price'] = (data['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(data))).round(2)
    
    # --- USER INPUTS ---
    st.subheader("Set Daily Nutrition Targets")
    col1, col2, col3 = st.columns(3)
    cal_target = col1.number_input("Target Calories", value=2000, step=100)
    protein_target = col2.number_input("Target Protein (g)", value=75)
    fat_target = col3.number_input("Target Fat (g)", value=70)

    st.subheader("Evolutionary Programming Parameters")
    col_g, col_p = st.columns(2)
    POP_SIZE = col_p.number_input("Population Size", value=80, step=10)
    GENS = col_g.number_input("Number of Generations", value=100, step=10)

    # --- POOLS ---
    b_pool = data[['Breakfast Suggestion','Calories','Protein','Fat','Price']].dropna()
    l_pool = data[['Lunch Suggestion','Calories','Protein','Fat','Price']].dropna()
    d_pool = data[['Dinner Suggestion','Calories','Protein','Fat','Price']].dropna()
    s_pool = data[['Snack Suggestion','Calories','Protein','Fat','Price']].dropna()

    # --- FITNESS FUNCTION ---
    def fitness(meals):
        t_cal = sum(m['Calories'] for m in meals)
        t_prot = sum(m['Protein'] for m in meals)
        t_fat = sum(m['Fat'] for m in meals)
        t_price = sum(m['Price'] for m in meals)

        penalty = 0
        # Hard constraints
        if t_cal < 1200 or t_cal > 3000:
            penalty += 100000
        if t_prot < protein_target * 0.8 or t_prot > protein_target * 1.5:
            penalty += 50000
        if t_fat < fat_target * 0.8 or t_fat > fat_target * 1.5:
            penalty += 50000

        # Soft constraints
        penalty += abs(t_cal - cal_target) * 10
        penalty += abs(t_prot - protein_target) * 5
        penalty += abs(t_fat - fat_target) * 5

        return t_price + penalty, t_cal, t_prot, t_fat, t_price

    # --- EP ALGORITHM ---
    def run_ep():
        # Initialize population
        pop = [[b_pool.sample(n=1).iloc[0], l_pool.sample(n=1).iloc[0],
                d_pool.sample(n=1).iloc[0], s_pool.sample(n=1).iloc[0]] for _ in range(POP_SIZE)]
        history = []

        for gen in range(GENS):
            scores = [fitness(ind)[0] for ind in pop]
            history.append(min(scores))

            # Select best half
            idx_sorted = np.argsort(scores)
            pop = [pop[i] for i in idx_sorted[:POP_SIZE//2]]

            # Mutation: change 1 random meal
            children = []
            for parent in pop:
                child = list(parent)
                idx = np.random.randint(0, 4)
                pools = [b_pool, l_pool, d_pool, s_pool]
                child[idx] = pools[idx].sample(n=1).iloc[0]
                children.append(child)
            pop.extend(children)

        final_scores = [fitness(ind)[0] for ind in pop]
        best_idx = np.argmin(final_scores)
        return pop[best_idx], history

    # --- RUN OPTIMIZER ---
    if st.button("🚀 Find Optimized Meal Plan"):
        winner, history = run_ep()
        _, f_cal, f_prot, f_fat, f_price = fitness(winner)

        # --- SHOW RESULTS ---
        st.divider()
        st.subheader("Optimized Daily Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cost (RM)", f"{f_price:.2f}")
        c2.metric("Total Calories", f"{f_cal} kcal")
        c3.metric("Total Protein", f"{f_prot} g")
        c4.metric("Total Fat", f"{f_fat} g")

        st.write("### 📋 Recommended Menu")
        labels = ["Breakfast","Lunch","Dinner","Snack"]
        cols = ["Breakfast Suggestion","Lunch Suggestion","Dinner Suggestion","Snack Suggestion"]
        for i in range(4):
            st.success(f"**{labels[i]}:** {winner[i][cols[i]]} ({winner[i]['Calories']} kcal) - RM {winner[i]['Price']:.2f}")

        # Convergence chart
        st.subheader("Convergence Chart")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(history, color='purple')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Score")
        st.pyplot(fig)

else:
    st.error("Please make sure 'Food_and_Nutrition__.csv' is in this folder.")
