"""Warehouse lease optimizer built with Streamlit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog


st.set_page_config(page_title="Warehouse Lease Optimizer", page_icon="📦", layout="wide")


def solve_lease_problem(requirements: list[float], period_costs: list[float]):
    """Minimize leasing cost while covering each month's requirement.

    A decision variable represents square feet leased from a starting month
    through an ending month. The cost is the per-square-foot price for that
    lease's duration.
    """
    months = len(requirements)
    variables: list[tuple[int, int, int]] = []
    objective: list[float] = []

    for start in range(months):
        for end in range(start, months):
            duration = end - start + 1
            variables.append((start, end, duration))
            objective.append(period_costs[duration - 1])

    # linprog uses A_ub x <= b_ub. Coverage constraints are naturally >=.
    coverage = np.array(
        [[-float(start <= month <= end) for start, end, _ in variables]
         for month in range(months)]
    )

    result = linprog(
        c=np.array(objective),
        A_ub=coverage,
        b_ub=-np.array(requirements),
        bounds=(0, None),
        method="highs",
    )

    if not result.success:
        raise ValueError(result.message)

    leases = []
    for (start, end, duration), square_feet in zip(variables, result.x):
        if square_feet > 0.005:
            leases.append(
                {
                    "Start month": start + 1,
                    "End month": end + 1,
                    "Lease duration": duration,
                    "Square feet": square_feet,
                    "Cost per sq. ft.": period_costs[duration - 1],
                    "Lease cost": square_feet * period_costs[duration - 1],
                }
            )

    selected = pd.DataFrame(leases)
    actual_coverage = -coverage @ result.x
    monthly = pd.DataFrame(
        {
            "Month": np.arange(1, months + 1),
            "Required space": requirements,
            "Leased space": actual_coverage,
        }
    )
    monthly["Surplus space"] = monthly["Leased space"] - monthly["Required space"]
    return result.fun, selected, monthly


st.title("📦 Warehouse Lease Optimizer")
st.write(
    "Find the least-cost combination of warehouse leases. A lease can start in "
    "any month, last for one or more consecutive months, and cover at least the "
    "required space during its term."
)

with st.sidebar:
    st.header("Problem setup")
    month_count = st.number_input("Number of months", min_value=1, max_value=24, value=5, step=1)

    default_requirements = [30000, 20000, 40000, 10000, 50000]
    default_costs = [65, 100, 135, 160, 190]
    requirements = []
    period_costs = []

    st.subheader("Inputs")
    for month in range(int(month_count)):
        default = default_requirements[month] if month < len(default_requirements) else 0
        requirements.append(
            st.number_input(
                f"Month {month + 1} space (sq. ft.)",
                min_value=0.0,
                value=float(default),
                step=1000.0,
                key=f"requirement_{month}",
            )
        )

    st.caption("Lease prices are total prices per square foot for the full lease period.")
    for duration in range(int(month_count)):
        default = default_costs[duration] if duration < len(default_costs) else 0
        period_costs.append(
            st.number_input(
                f"{duration + 1}-month lease ($/sq. ft.)",
                min_value=0.0,
                value=float(default),
                step=1.0,
                key=f"cost_{duration}",
            )
        )

if any(cost < 0 for cost in period_costs):
    st.error("Lease costs cannot be negative.")
else:
    total_cost, leases, monthly = solve_lease_problem(requirements, period_costs)

    st.metric("Minimum total leasing cost", f"${total_cost:,.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Optimal lease plan")
        display_leases = leases.copy()
        if not display_leases.empty:
            display_leases["Square feet"] = display_leases["Square feet"].round(0).map(lambda x: f"{x:,.0f}")
            display_leases["Cost per sq. ft."] = display_leases["Cost per sq. ft."].map(lambda x: f"${x:,.2f}")
            display_leases["Lease cost"] = display_leases["Lease cost"].map(lambda x: f"${x:,.2f}")
            st.dataframe(display_leases, hide_index=True, use_container_width=True)
        else:
            st.info("No space is required, so no leases are needed.")

    with right:
        st.subheader("Monthly coverage")
        chart_data = monthly.set_index("Month")[["Required space", "Leased space"]]
        st.bar_chart(chart_data)
        st.dataframe(
            monthly.style.format(
                {
                    "Required space": "{:,.0f}",
                    "Leased space": "{:,.0f}",
                    "Surplus space": "{:,.0f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("How the optimization works"):
        st.markdown(
            "The app creates one decision variable for every possible lease "
            "interval. It then minimizes the sum of lease costs subject to "
            "leased space being at least the requirement in every month. "
            "Because space is divisible, the solution can combine leases for "
            "different amounts of square footage."
        )
