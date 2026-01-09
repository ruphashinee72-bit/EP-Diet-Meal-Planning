import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Diet Optimizer RM", layout="wide")
st.title("🍱 Final Diet Optimizer (RM)")

# 1. LOAD AND CLEAN DATA (This stops errors)
if os.path.exists("Food_and_Nutrition__.csv"):
    df = pd.read_csv("Food_and_Nutrition__.csv")
    # This line removes any hidden spaces in your column names!
    df.columns = df.columns.str.strip() 
    
    # Generate prices in RM
    np.random.seed(42)
    df['Price'] = (df['Calories'] * 0.005 + np.random.uniform(2, 5, size=len(df))).round(2)

    # 2. TARGETS
    st.subheader("Set Daily Nutrition Targets")
    c1, c2, c3 = st.columns(3)
    cal_target = c1.number_input("Target Calories", 1200, 3000, 2000)
    prot_target = c2.number_input("Target Protein (g)", 10, 300, 75)
    fat_target = c3.number_input("Target Fat (g)", 10, 200, 70)

    # 3. THE FITNESS BRAIN
    def get_fitness(meals):
        t_cal = sum(m['Calories'] for m in meals)
        t_prot = sum(m['Protein'] for m in meals)
        t_fat = sum(m['Fat'] for m in meals)
        t_price = sum(m['Price'] for m in meals)

        # THE HARD RULES:
        penalty = 0
        if t_cal < 1200: penalty += 50000 + (1200 - t_cal) * 100
        if t_cal > 3000: penalty += 50000 + (t_cal - 3000) * 100
        
        # CLOSE TO TARGET RULES:
        penalty += abs(t_cal - cal_target) * 10
        penalty += abs(t_prot - prot_target) * 5
        penalty += abs(t_fat - fat_target) * 5
        
        return t_price + penalty, t_cal, t_prot, t_fat, t_price

    # 4. RUN OPTIMIZER
    if st.button("🚀 RUN OPTIMIZER NOW"):
        # Create Pools
        b_p = df[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        l_p = df[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        d_p = df[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
        s_p = df[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()

        # Initial Pop
        pop = [[b_p.sample(1).iloc[0], l_p.sample(1).iloc[0], 
                d_p.sample(1).iloc[0], s_p.sample(1).iloc[0]] for _ in range(60)]
        
        history = []
        for gen in range(100):
            pop.sort(key=lambda x: get_fitness(x)[0])
            history.append(get_fitness(pop[0])[0])
            
            # Keep best, Mutate others
            new_pop = pop[:30]
            while len(new_pop) < 60:
                parent = pop[np.random.randint(0, 30)]
                child = list(parent)
                idx = np.random.randint(0, 4)
                child[idx] = [b_p, l_p, d_p, s_p][idx].sample(1).iloc[0]
                new_pop.append(child)
            pop = new_pop
        
        winner = pop[0]
        _, f_cal, f_prot, f_fat, f_price = get_fitness(winner)

        # 5. RESULTS
        st.divider()
        res1, res2, res3, res4 = st.columns(4)
        res1.metric("TOTAL COST", f"RM {f_price:.2f}")
        res2.metric("CALORIES", f"{f_cal} kcal")
        res3.metric("PROTEIN", f"{f_prot}g")
        res4.metric("FAT", f"{f_fat}g")

        st.subheader("🍴 Recommended Menu")
        names = ["Breakfast", "Lunch", "Dinner", "Snack"]
        cols = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        for i in range(4):
            st.success(f"**{names[i]}:** {winner[i][cols[i]]} ({winner[i]['Calories']} kcal) - RM {winner[i]['Price']}")
            
        st.line_chart(history)
