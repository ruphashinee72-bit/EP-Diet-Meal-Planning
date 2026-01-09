import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer RM", layout="wide")
st.title("🥗 Final Diet Optimizer: Strict Constraint Mode")

if os.path.exists("Food_and_Nutrition__.csv"):
    df = pd.read_csv("Food_and_Nutrition__.csv")
    df.columns = df.columns.str.strip() 
    
    # Prices in RM
    np.random.seed(42)
    df['Price'] = (df['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(df))).round(2)

    # --- SIDEBAR TARGETS ---
    st.sidebar.header("Step 1: Set Constraints")
    cal_max = 3000
    cal_min = 1200
    target_cal = st.sidebar.slider("Target Calories", cal_min, cal_max, 2000)
    min_prot = st.sidebar.number_input("Min Protein (g)", 50)

    # --- THE ENGINE ---
    def get_stats(meals):
        c = sum(m['Calories'] for m in meals)
        p = sum(m['Protein'] for m in meals)
        f = sum(m['Fat'] for m in meals)
        cost = sum(m['Price'] for m in meals)
        
        # PENALTY LOGIC
        # If outside 1200-3000, penalty is massive
        penalty = 0
        if c > cal_max: penalty += (c - cal_max) * 500 + 10000
        if c < cal_min: penalty += (cal_min - c) * 500 + 10000
        
        # Soft penalty for missing the specific target
        penalty += abs(c - target_cal) * 20
        if p < min_prot: penalty += (min_prot - p) * 100
        
        return cost + penalty, c, p, f, cost

    def evolve():
        # Create pools
        pools = [
            df[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
            df[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
            df[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
            df[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        ]

        pop_size = 100
        # Initialize with totally random individuals
        pop = [[p.sample(1).iloc[0] for p in pools] for _ in range(pop_size)]
        
        history = []
        for gen in range(200): # More generations to search harder
            # Evaluate all
            evals = [get_stats(ind) for ind in pop]
            fitness_scores = [e[0] for e in evals]
            history.append(min(fitness_scores))
            
            # Selection: Tournament (Pick 3, keep the best)
            new_pop = []
            for _ in range(pop_size // 2):
                participants = np.random.choice(len(pop), 3, replace=False)
                best_idx = participants[np.argmin([fitness_scores[i] for i in participants])]
                new_pop.append(list(pop[best_idx]))

            # Reproduction with High Mutation
            while len(new_pop) < pop_size:
                parent = new_pop[np.random.randint(0, len(new_pop))]
                child = [m.copy() for m in parent]
                
                # High Mutation: 50% chance to swap a meal
                if np.random.rand() < 0.5:
                    m_idx = np.random.randint(0, 4)
                    child[m_idx] = pools[m_idx].sample(1).iloc[0]
                new_pop.append(child)
            
            pop = new_pop
            
        # Final winner
        winner_evals = [get_stats(ind) for ind in pop]
        winner_idx = np.argmin([e[0] for e in winner_evals])
        return pop[winner_idx], history

    if st.button("🚀 RUN OPTIMIZATION"):
        with st.spinner("AI is searching for low-cost, low-calorie meals..."):
            winner, history = evolve()
            fit, final_cal, final_prot, final_fat, final_cost = get_stats(winner)

        # --- RESULTS ---
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("TOTAL COST", f"RM {final_cost:.2f}")
        col2.metric("CALORIES", f"{final_cal} kcal")
        col3.metric("PROTEIN", f"{final_prot}g")
        col4.metric("FAT", f"{final_fat}g")

        if final_cal > 3000:
            st.error("🚨 Constraints Failed: Still too high. Try clicking Run again to restart the search.")
        else:
            st.success("✅ Constraints Satisfied: Found a meal plan under 3000 kcal!")

        st.subheader("📋 Your Optimized Daily Menu")
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        for i in range(4):
            st.info(f"**{labels[i]}**: {winner[i][cols[i]]} ({winner[i]['Calories']} kcal) - RM {winner[i]['Price']}")

        st.line_chart(history)
else:
    st.error("CSV File not found.")
