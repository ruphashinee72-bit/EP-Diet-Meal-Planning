import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Evolutionary Diet Optimizer", layout="wide")
st.title("🥗 Evolutionary Diet Optimizer (RM)")
st.write("Objective: Minimize Total Cost while satisfying 1200-3000 kcal constraints.")

# --- 1. LOAD & CLEAN DATA ---
if os.path.exists("Food_and_Nutrition__.csv"):
    df = pd.read_csv("Food_and_Nutrition__.csv")
    df.columns = df.columns.str.strip() # Remove hidden spaces
    
    # Generate Synthetic Price in RM (Objective: Minimize this!)
    np.random.seed(42)
    df['Price'] = (df['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(df))).round(2)

    # --- 2. USER INPUTS ---
    st.sidebar.header("Nutrition Constraints")
    cal_target = st.sidebar.slider("Calorie Goal", 1200, 3000, 2000)
    prot_min = st.sidebar.number_input("Min Protein (g)", value=50)
    fat_min = st.sidebar.number_input("Min Fat (g)", value=40)

    # --- 3. THE EVOLUTIONARY ENGINE ---
    def fitness_function(meals):
        t_cal = sum(m['Calories'] for m in meals)
        t_prot = sum(m['Protein'] for m in meals)
        t_fat = sum(m['Fat'] for m in meals)
        t_price = sum(m['Price'] for m in meals)

        # START WITH THE COST (Objective)
        score = t_price 

        # ADD HEAVY PENALTIES FOR BREAKING CONSTRAINTS
        # Penalty for being outside 1200 - 3000 range
        if t_cal < 1200 or t_cal > 3000:
            score += 5000  # Massive penalty
        
        # Penalty for missing specific targets
        score += abs(t_cal - cal_target) * 5
        if t_prot < prot_min: score += (prot_min - t_prot) * 20
        if t_fat < fat_min: score += (fat_min - t_fat) * 20
        
        return score, t_cal, t_prot, t_fat, t_price

    def run_evolution():
        # Separate meals into pools
        b_pool = df[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        l_pool = df[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        d_pool = df[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        s_pool = df[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()

        # Initial Population (60 random plans)
        pop = [[b_pool.sample(1).iloc[0], l_pool.sample(1).iloc[0], 
                d_pool.sample(1).iloc[0], s_pool.sample(1).iloc[0]] for _ in range(60)]
        
        history = []
        for gen in range(100):
            # Sort by fitness score (Lower is better)
            pop.sort(key=lambda x: fitness_function(x)[0])
            history.append(fitness_function(pop[0])[0])
            
            # Selection & Mutation
            next_gen = pop[:20] # Keep top 20 survivors
            while len(next_gen) < 60:
                parent = next_gen[np.random.randint(0, len(next_gen))]
                child = list(parent)
                # Mutate: Change 1 random meal
                idx = np.random.randint(0, 4)
                child[idx] = [b_pool, l_pool, d_pool, s_pool][idx].sample(1).iloc[0]
                next_gen.append(child)
            pop = next_gen
            
        return pop[0], history

    # --- 4. EXECUTION ---
    if st.button("🚀 Optimize My Plan"):
        with st.spinner("Evolution in progress..."):
            winner, history = run_evolution()
            _, f_cal, f_prot, f_fat, f_price = fitness_function(winner)

        # DISPLAY RESULTS
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MINIMIZED COST", f"RM {f_price:.2f}")
        c2.metric("TOTAL CALORIES", f"{f_cal} kcal")
        c3.metric("TOTAL PROTEIN", f"{f_prot}g")
        c4.metric("TOTAL FAT", f"{f_fat}g")

        if f_cal > 3000 or f_cal < 1200:
            st.error("Constraint Violation: The AI could not find a plan in range. Please try again.")
        else:
            st.success("Target Constraints Satisfied!")

        st.subheader("📋 Recommended Meal Plan")
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        for i in range(4):
            st.write(f"**{labels[i]}:** {winner[i][cols[i]]} | RM {winner[i]['Price']}")

        st.subheader("📈 Optimization Progress")
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(history, color='green')
        ax.set_title("Fitness Score Over Generations")
        st.pyplot(fig)
else:
    st.error("CSV file not found. Please upload it to your GitHub.")
