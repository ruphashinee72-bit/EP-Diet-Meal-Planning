import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer RM", layout="wide")
st.title("🍱 Diet Meal Planning (Strict Constraints)")

if os.path.exists("Food_and_Nutrition__.csv"):
    data = pd.read_csv("Food_and_Nutrition__.csv")
    
    # --- Generate Synthetic Price (RM) ---
    np.random.seed(42)
    data['Price'] = (data['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(data))).round(2)

    # --- USER INPUTS ---
    st.subheader("Set Daily Nutrition Targets")
    col_in1, col_in2, col_in3 = st.columns(3)
    cal_target = col_in1.number_input("Target Calories", value=2000, step=50)
    protein_target = col_in2.number_input("Protein (g)", value=75, step=5)
    fat_target = col_in3.number_input("Fat (g)", value=70, step=5)

    st.subheader("EP Parameters")
    col_ep1, col_ep2 = st.columns(2)
    POP_SIZE = col_ep1.number_input("Population Size", value=50, step=10)
    GENS = col_ep2.number_input("Generations", value=50, step=10)

    # --- Define Pools ---
    b_pool = data[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    l_pool = data[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    d_pool = data[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    s_pool = data[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()

    # --- FITNESS FUNCTION ---
    def fitness(meals):
        t_cal = sum(m['Calories'] for m in meals)
        t_prot = sum(m['Protein'] for m in meals)
        t_fat = sum(m['Fat'] for m in meals)
        t_price = sum(m['Price'] for m in meals)

        # Penalty for breaking limits
        penalty = 0
        if t_cal < 1200 or t_cal > 3000:
            penalty += 1e5
        # Soft penalties for deviation from targets
        penalty += abs(t_cal - cal_target) * 10
        penalty += abs(t_prot - protein_target) * 5
        penalty += abs(t_fat - fat_target) * 5

        return t_price + penalty

    # --- INITIAL POPULATION: guided random (close to targets) ---
    def init_population():
        pop = []
        for _ in range(POP_SIZE):
            while True:
                b = b_pool.sample(n=1).iloc[0]
                l = l_pool.sample(n=1).iloc[0]
                d = d_pool.sample(n=1).iloc[0]
                s = s_pool.sample(n=1).iloc[0]
                total_cal = b['Calories'] + l['Calories'] + d['Calories'] + s['Calories']
                # Accept only roughly near target to reduce extreme overshoot
                if 1200 <= total_cal <= 3000:
                    pop.append([b, l, d, s])
                    break
        return pop

    # --- MUTATION: swap one meal for another in its pool ---
    def mutate(individual):
        child = individual.copy()
        idx = np.random.randint(0, 4)
        pools = [b_pool, l_pool, d_pool, s_pool]
        child[idx] = pools[idx].sample(n=1).iloc[0]
        return child

    # --- EP OPTIMIZER ---
    def run_ep():
        pop = init_population()
        history = []

        for gen in range(GENS):
            # Compute fitness
            scores = [fitness(ind) for ind in pop]
            history.append(min(scores))

            # Select top half
            idx_sorted = np.argsort(scores)
            pop = [pop[i] for i in idx_sorted[:POP_SIZE//2]]

            # Generate children via mutation
            children = [mutate(parent) for parent in pop]
            pop.extend(children)

        # Final selection
        final_scores = [fitness(ind) for ind in pop]
        best_idx = np.argmin(final_scores)
        return pop[best_idx], history

    if st.button("🚀 Find My Optimized Meal Plan"):
        winner, history = run_ep()

        total_cal = sum(m['Calories'] for m in winner)
        total_prot = sum(m['Protein'] for m in winner)
        total_fat = sum(m['Fat'] for m in winner)
        total_price = sum(m['Price'] for m in winner)

        # --- SHOW RESULTS ---
        st.divider()
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
            st.success(f"**{labels[i]}:** {winner[i][cols[i]]} ({winner[i]['Calories']} kcal) - RM {winner[i]['Price']:.2f}")

        st.subheader("Convergence Chart")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, color='purple')
        ax.set_ylabel("Fitness (Lower is Better)")
        ax.set_xlabel("Generation")
        st.pyplot(fig)

else:
    st.error("Please ensure 'Food_and_Nutrition__.csv' is in the folder.")
