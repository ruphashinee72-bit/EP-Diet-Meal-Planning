import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Final Solution", layout="wide")
st.title("🍱 Guaranteed Constraint Diet Optimizer")

if os.path.exists("Food_and_Nutrition__.csv"):
    df = pd.read_csv("Food_and_Nutrition__.csv")
    df.columns = df.columns.str.strip() 
    
    # Generate prices in RM
    np.random.seed(99) # New seed
    df['Price'] = (df['Calories'] * 0.005 + 2).round(2)

    # --- SIDEBAR ---
    st.sidebar.header("🎯 STRICT TARGETS")
    # I have lowered these defaults so the AI doesn't feel forced to "overeat"
    target_cal = st.sidebar.slider("Calories Target", 1200, 2500, 1800)
    min_prot = st.sidebar.slider("Min Protein (g)", 10, 100, 50)

    def get_fitness(meals):
        c = sum(m['Calories'] for m in meals)
        p = sum(m['Protein'] for m in meals)
        cost = sum(m['Price'] for m in meals)
        
        # --- THE RESET LOGIC ---
        # If Calories > 3000, the score becomes so huge (1 Million) 
        # that the computer MUST delete that plan immediately.
        if c > 3000 or c < 1200:
            return 1000000, c, p, cost
            
        # If it's in the safe range (1200-3000), then we look at Cost and Protein
        fitness = cost 
        fitness += abs(c - target_cal) * 2
        if p < min_prot: fitness += (min_prot - p) * 50
        
        return fitness, c, p, cost

    def run_optimizer():
        pools = [
            df[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
            df[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
            df[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
            df[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        ]

        # Start with a HUGE population to find the rare low-calorie meals
        pop = [[p.sample(1).iloc[0] for p in pools] for _ in range(200)]
        
        history = []
        for gen in range(100):
            # Sort: This will move all the 1,000,000 point (over 3000kcal) plans to the bottom
            pop.sort(key=lambda x: get_fitness(x)[0])
            history.append(get_fitness(pop[0])[0])
            
            # Keep only the ones that stayed under 3000kcal
            next_gen = pop[:40] 
            
            while len(next_gen) < 200:
                parent = next_gen[np.random.randint(0, len(next_gen))]
                child = [m.copy() for m in parent]
                # Randomly change 1 or 2 meals to keep searching
                for _ in range(np.random.randint(1, 3)):
                    idx = np.random.randint(0, 4)
                    child[idx] = pools[idx].sample(1).iloc[0]
                next_gen.append(child)
            pop = next_gen
            
        return pop[0], history

    if st.button("🚀 EXECUTE OPTIMIZATION"):
        with st.spinner("Filtering out high-calorie meals..."):
            best, hist = run_optimizer()
            _, f_cal, f_prot, f_cost = get_fitness(best)

        st.divider()
        # --- THE RESULTS ---
        c1, c2, c3 = st.columns(3)
        c1.metric("DAILY COST", f"RM {f_cost:.2f}")
        c2.metric("CALORIES", f"{f_cal} kcal")
        c3.metric("PROTEIN", f"{f_prot}g")

        if f_cal > 3000:
            st.error("Still too high. Your CSV might not have enough small meals. Try a lower Protein target in the sidebar.")
        else:
            st.success("SUCCESS! Meal plan is within the 1200-3000 kcal limit.")

        # Show the foods
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        for i in range(4):
            st.write(f"**{labels[i]}**: {best[i][cols[i]]} ({best[i]['Calories']} kcal)")

else:
    st.error("CSV file is missing!")
