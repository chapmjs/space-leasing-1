import streamlit as st
import pulp
import pandas as pd

st.title("📦 Web Mercantile Warehouse Leasing Optimizer")
st.subheader("Version 1.0 - Baseline Model")

# 1. Define Hardcoded Data
months = [1, 2, 3, 4, 5]
req_space = {1: 30000, 2: 20000, 3: 40000, 4: 10000, 5: 50000}
lease_cost = {1: 65, 2: 100, 3: 135, 4: 160, 5: 190}

# Display Baseline Problem Setup
col1, col2 = st.columns(2)
with col1:
    st.write("**Space Requirements (Sq. Ft.)**")
    st.dataframe(pd.DataFrame(list(req_space.items()), columns=["Month", "Required Space"]))

with col2:
    st.write("**Leasing Costs ($ / Sq. Ft.)**")
    st.dataframe(pd.DataFrame(list(lease_cost.items()), columns=["Duration (Months)", "Cost ($)"]))

# 2. Formulate Linear Program
prob = pulp.LpProblem("Web_Mercantile_Min_Cost", pulp.LpMinimize)

# Decision Variables x[i, j]
x = {}
for i in months:
    for j in range(1, 6 - i + 1):
        x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous')

# Objective Function
prob += pulp.lpSum(x[i, j] * lease_cost[j] for (i, j) in x), "Total_Leasing_Cost"

# Constraints
for t in months:
    prob += (
        pulp.lpSum(x[i, j] for (i, j) in x if i <= t and i + j - 1 >= t) >= req_space[t],
        f"Requirement_Month_{t}"
    )

# 3. Solve Model
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# 4. Output Results
st.markdown("---")
st.header("Optimization Results")

if pulp.LpStatus[prob.status] == "Optimal":
    min_cost = pulp.value(prob.objective)
    st.success(f"**Optimal Total Leasing Cost:** ${min_cost:,.2f}")

    # Format output schedule
    results = []
    for (i, j), var in x.items():
        val = var.varValue
        if val > 0:
            results.append({
                "Start Month": i,
                "Lease Duration (Months)": j,
                "End Month": i + j - 1,
                "Leased Space (Sq. Ft.)": f"{val:,.0f}",
                "Cost per Sq. Ft.": f"${lease_cost[j]}",
                "Total Cost": f"${val * lease_cost[j]:,.2f}"
            })
    
    st.subheader("Optimal Leasing Plan")
    st.table(pd.DataFrame(results))
else:
    st.error("No optimal solution found.")
