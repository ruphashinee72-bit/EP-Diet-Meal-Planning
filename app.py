import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer RM", layout="wide")
st.title("🍱 Diet Meal Planning (Strict Constraints)")

if os.path.exists("Food_and_Nutrition__.csv"):
    data = pd.read_csv("Food_and_Nutrition__.csv")
    
    # Generate Synthetic Price in RM
    np.random.seed(42)
    data['Price'] = (data['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(data))).round(2)

    # --- STEP 1: USER INPUTS ---
    st.subheader("Set Daily Nutrition Targets")
    col_in1, col_in2, col_in3 = st.columns(3)
    cal_target = col_in1.number_input("Target Calories", value=2000, step=100)
    protein_target = col_in2.number_input("Protein (g)", value=75)
    fat_target = col_in3.number_input("Fat (g)", value=70)

    # Pools
    b_pool = data[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    l_pool = data[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    d_pool = data[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    s_pool = data[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()

    # --- THE FIXED FITNESS FUNCTION (Strict Range) ---
    def fitness(meals):
        t_cal = sum(m['Calories'] for m in meals)
        t_prot = sum(m['Protein'] for m in meals)
        t_fat = sum(m['Fat'] for m in meals)
        t_price = sum(m['Price'] for m in meals)

        # 1. HARD CONSTRAINT: Must be between 1200 and 3000
        penalty = 0
        if t_cal < 1200 or t_cal > 3000:
            penalty += 100000  # Massive penalty for breaking your rule
        
        # 2. SOFT CONSTRAINT: Accuracy to the specific target
        penalty += abs(t_cal - cal_target) * 10
        penalty += abs(t_prot - protein_target) * 5
        penalty += abs(t_fat - fat_target) * 5
        
        # We add the Price. Lower score = Better plan.
        return t_price + penalty, t_cal, t_prot, t_fat, t_price

    def run_ep():
        POP_SIZE = 80
        GENS = 100
        
        # Initial random population
        pop = [[b_pool.sample(n=1).iloc[0], l_pool.sample(n=1).iloc[0], 
                d_pool.sample(n=1).iloc[0], s_pool.sample(n=1).iloc[0]] for _ in range(POP_SIZE)]
        
        history = []
        for gen in range(GENS):
            # Sort by fitness (lowest score first)
            scores = [fitness(ind)[0] for ind in pop]
            history.append(min(scores))
            
            idx_sorted = np.argsort(scores)
            pop = [pop[i] for i in idx_sorted[:POP_SIZE//2]]
            
            # Mutation (Change 1 meal)
            children = []
            for parent in pop:
                child = list(parent)
                idx = np.random.randint(0, 4)
                pools = [b_pool, l_pool, d_pool, s_pool]
                child[idx] = pools[idx].sample(n=1).iloc[0]
                children.append(child)
            pop.extend(children)
            
        final_scores = [fitness(ind)[0] for ind in pop]
        return pop[np.argmin(final_scores)], history

    if st.button("🚀 Find My Optimized Meal Plan"):
        winner, history = run_ep()
        _, f_cal, f_prot, f_fat, f_price = fitness(winner)

        # --- VALIDATION CHECK ---
        if f_cal < 1200 or f_cal > 3000:
            st.error(f"Warning: Result ({f_cal} kcal) is outside your 1200-3000 range. Try running again or changing targets.")
        
        st.divider()
        st.subheader("Optimized Daily Result (RM)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Daily Cost", f"RM {f_price:.2f}")
        m2.metric("Total Calories", f"{f_cal} kcal")
        m3.metric("Total Protein", f"{f_prot}g")
        m4.metric("Total Fat", f"{f_fat}g")

        st.write("### 📋 Recommended Menu")
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        for i in range(4):
            st.success(f"**{labels[i]}:** {winner[i][cols[i]]} ({winner[i]['Calories']} kcal) - RM {winner[i]['Price']:.2f}")

        st.subheader("Convergence Chart")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, color='purple')
        ax.set_ylabel("Penalty Score")
        ax.set_xlabel("Generation")
        st.pyplot(fig)
else:
    st.error("Please ensure 'Food_and_Nutrition__.csv' is in the folder.")
