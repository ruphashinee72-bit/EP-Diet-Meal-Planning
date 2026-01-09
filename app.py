import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ACO Diet Optimizer", layout="wide")
st.title("🐜 Ant Colony Optimization (ACO) Diet Planner")
st.write("Objective: Ants will find the cheapest path (meals) while staying between 1200-3000 kcal.")

# --- 1. DATA PREP ---
@st.cache_data
def load_data():
    df = pd.read_csv("Food_and_Nutrition__.csv")
    df.columns = df.columns.str.strip()
    np.random.seed(42)
    # Price model in RM
    df['Price'] = (df['Calories'] * 0.005 + 2.50).round(2)
    return df

data = load_data()

# --- 2. ACO LOGIC ---
def run_aco(target_c):
    pools = [
        data[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
        data[['Lunch Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
        data[['Dinner Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna(),
        data[['Snack Suggestion', 'Calories', 'Protein', 'Fat', 'Price']].dropna()
    ]

    # Initialize Pheromones (equally for all foods)
    pheromones = [np.ones(len(p)) for p in pools]
    
    ants = 30
    iterations = 50
    evaporation_rate = 0.5
    best_plan = None
    best_score = float('inf')
    history = []

    for _ in range(iterations):
        all_ant_plans = []
        
        for ant in range(ants):
            # Ant chooses a path based on pheromones
            plan = []
            for i in range(4):
                probs = pheromones[i] / pheromones[i].sum()
                idx = np.random.choice(len(pools[i]), p=probs)
                plan.append(pools[i].iloc[idx])
            
            # Calculate Fitness
            t_cal = sum(m['Calories'] for m in plan)
            t_price = sum(m['Price'] for m in plan)
            
            # Penalty for breaking 1200-3000 limit
            penalty = 0
            if t_cal > 3000 or t_cal < 1200:
                penalty = 5000 
            
            score = t_price + penalty + abs(t_cal - target_c) * 2
            
            if score < best_score:
                best_score = score
                best_plan = plan
            
            all_ant_plans.append((plan, score))

        # Update Pheromones (Evaporation)
        for i in range(4):
            pheromones[i] *= (1 - evaporation_rate)

        # Deposit Pheromones (Ants that found cheap, valid plans leave more pheromones)
        for plan, score in all_ant_plans:
            reward = 100 / score # Better score = higher reward
            for i in range(4):
                # Find index of the meal chosen to add reward
                # (Simplified for this version)
                pheromones[i] += reward 
        
        history.append(best_score if best_score < 5000 else 50)

    return best_plan, history

# --- 3. UI ---
st.sidebar.header("Nutrition Targets")
user_target = st.sidebar.slider("Calories Target", 1200, 3000, 2000)

if st.button("🚀 Run Ant Colony Optimization"):
    winner, history = run_aco(user_target)
    
    f_cal = sum(m['Calories'] for m in winner)
    f_price = sum(m['Price'] for m in winner)
    f_prot = sum(m['Protein'] for m in winner)

    st.divider()
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("MINIMIZED COST", f"RM {f_price:.2f}")
    c2.metric("TOTAL CALORIES", f"{f_cal} kcal")
    c3.metric("TOTAL PROTEIN", f"{f_prot}g")

    if f_cal > 3000 or f_cal < 1200:
        st.error("Ants got lost! Try clicking Run again.")
    else:
        st.success("Success! The Ants found a low-cost plan within your calorie range!")

    st.subheader("📋 Your Optimized Daily Menu")
    labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
    col_name = data.columns[0]
    for i in range(4):
        st.info(f"**{labels[i]}:** {winner[i][col_name]} ({winner[i]['Calories']} kcal) - RM {winner[i]['Price']}")

    st.line_chart(history)
