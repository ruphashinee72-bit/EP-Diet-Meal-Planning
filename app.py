import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer RM", layout="wide")
st.title("🍱 Diet Meal Planning Optimisation (Evolutionary Programming)")

if os.listdir(): # Checks for CSV
    # Load and clean data
    data = pd.read_csv("Food_and_Nutrition__.csv")
    np.random.seed(42)
    data['Price'] = (data['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(data))).round(2)

    st.subheader("Set Daily Nutrition Targets")
    col_in1, col_in2, col_in3 = st.columns(3)
    cal_target = col_in1.number_input("Calories", value=1000, step=100)
    protein_target = col_in2.number_input("Protein (g)", value=50, step=5)
    fat_target = col_in3.number_input("Fat (g)", value=40, step=5)

    # Pools
    b_pool = data[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    l_pool = data[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    d_pool = data[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    s_pool = data[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()

    # --- THE FIXED FITNESS FUNCTION (High Penalty) ---
    def fitness(meals):
        t_cal = sum(m['Calories'] for m in meals)
        t_prot = sum(m['Protein'] for m in meals)
        t_fat = sum(m['Fat'] for m in meals)
        t_price = sum(m['Price'] for m in meals)

        # WE MAKE CALORIES THE BOSS
        # abs(t_cal - cal_target) * 20 means if you are 100 kcal off, 
        # the penalty is 2000! This stops the 4550 kcal result.
        penalty = abs(t_cal - cal_target) * 20 
        penalty += abs(t_prot - protein_target) * 5
        penalty += abs(t_fat - fat_target) * 5
        
        return t_price + penalty, t_cal, t_prot, t_fat, t_price

    def run_ep():
        POP_SIZE = 100 # Increased population for better searching
        GENS = 150     # More generations to find the lower calorie meals
        
        pop = [[b_pool.sample(n=1).iloc[0], l_pool.sample(n=1).iloc[0], 
                d_pool.sample(n=1).iloc[0], s_pool.sample(n=1).iloc[0]] for _ in range(POP_SIZE)]
        
        history = []
        for gen in range(GENS):
            scores = [fitness(ind)[0] for ind in pop]
            history.append(min(scores))
            
            # Keep top 20% (Selection)
            idx_sorted = np.argsort(scores)
            pop = [pop[i] for i in idx_sorted[:POP_SIZE//5]]
            
            # Mutation (Fill the rest of the population with mutations)
            while len(pop) < POP_SIZE:
                parent = pop[np.random.randint(0, len(pop))]
                child = list(parent)
                idx = np.random.randint(0, 4)
                pools = [b_pool, l_pool, d_pool, s_pool]
                child[idx] = pools[idx].sample(n=1).iloc[0]
                pop.append(child)
                
        final_scores = [fitness(ind)[0] for ind in pop]
        return pop[np.argmin(final_scores)], history

    if st.button("🚀 Run Evolutionary Optimization"):
        winner, history = run_ep()
        _, f_cal, f_prot, f_fat, f_price = fitness(winner)

        st.divider()
        st.subheader("Optimized Result Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cost", f"RM {f_price:.2f}")
        # Delta will show you how much the AI missed by
        m2.metric("Total Calories", f"{f_cal} kcal", delta=f"{f_cal - cal_target} from target")
        m3.metric("Total Protein", f"{f_prot}g")
        m4.metric("Total Fat", f"{f_fat}g")

        st.write("### Recommended Daily Menu")
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        for i in range(4):
            st.success(f"**{labels[i]}:** {winner[i][cols[i]]} ({winner[i]['Calories']} kcal)")

        st.subheader("Performance Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, color='blue')
        ax.set_title("Evolutionary Convergence (Minimizing Calorie Error)")
        st.pyplot(fig)
else:
    st.error("Upload CSV")
