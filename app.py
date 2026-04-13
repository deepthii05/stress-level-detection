import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Stress Detector",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🧠 Stress Level Detection")
st.caption("A Fuzzy Logic Decision Support System for Stress Awareness")

st.info(
    "Scale Guide: 0 = Very Low / None • 5 = Moderate / Average • "
    "10 = Very High / Excellent depending on factor"
)

# -------------------------------------------------
# USER INPUTS
# -------------------------------------------------
st.subheader("Enter Your Details")

sleep_val = st.slider(
    "Sleep Hours", 0, 10, 5,
    help="0 = No sleep, 5 = Average sleep, 10 = Excellent sleep"
)

workload_val = st.slider(
    "Workload Level", 0, 10, 5,
    help="0 = No workload, 5 = Moderate, 10 = Extreme workload"
)

screen_val = st.slider(
    "Screen Time", 0, 10, 5,
    help="0 = None, 5 = Moderate use, 10 = Excessive screen time"
)

physical_val = st.slider(
    "Physical Activity", 0, 10, 5,
    help="0 = No exercise, 5 = Moderate, 10 = Highly active"
)

social_val = st.slider(
    "Social Interaction", 0, 10, 5,
    help="0 = Isolated, 5 = Normal, 10 = Highly social"
)

# -------------------------------------------------
# STRESS LEVEL GUIDE
# -------------------------------------------------
st.subheader("Stress Level Meaning")
st.success("0 - 3.99 : Low Stress")
st.warning("4 - 6.99 : Moderate Stress")
st.error("7 - 10 : High Stress")

# -------------------------------------------------
# FUZZY VARIABLES
# -------------------------------------------------
sleep = ctrl.Antecedent(np.arange(0, 11, 1), "sleep")
workload = ctrl.Antecedent(np.arange(0, 11, 1), "workload")
screen = ctrl.Antecedent(np.arange(0, 11, 1), "screen")
physical = ctrl.Antecedent(np.arange(0, 11, 1), "physical")
social = ctrl.Antecedent(np.arange(0, 11, 1), "social")

stress = ctrl.Consequent(np.arange(0, 11, 1), "stress")

sleep_rec = ctrl.Consequent(np.arange(0, 11, 1), "sleep_rec")
work_rec = ctrl.Consequent(np.arange(0, 11, 1), "work_rec")
activity_rec = ctrl.Consequent(np.arange(0, 11, 1), "activity_rec")
social_rec = ctrl.Consequent(np.arange(0, 11, 1), "social_rec")

# -------------------------------------------------
# MEMBERSHIP FUNCTIONS
# -------------------------------------------------
sleep["low"] = fuzz.trapmf(sleep.universe, [0, 0, 3, 5])
sleep["medium"] = fuzz.trimf(sleep.universe, [3, 5, 7])
sleep["high"] = fuzz.trapmf(sleep.universe, [6, 8, 10, 10])

workload["low"] = fuzz.trimf(workload.universe, [0, 0, 5])
workload["medium"] = fuzz.trimf(workload.universe, [3, 5, 7])
workload["high"] = fuzz.trimf(workload.universe, [5, 10, 10])

screen["low"] = fuzz.trapmf(screen.universe, [0, 0, 2, 4])
screen["medium"] = fuzz.trimf(screen.universe, [3, 5, 7])
screen["high"] = fuzz.trapmf(screen.universe, [6, 8, 10, 10])

physical["low"] = fuzz.trimf(physical.universe, [0, 0, 5])
physical["medium"] = fuzz.trimf(physical.universe, [3, 5, 7])
physical["high"] = fuzz.trimf(physical.universe, [5, 10, 10])

social["low"] = fuzz.gaussmf(social.universe, 2, 1.5)
social["medium"] = fuzz.gaussmf(social.universe, 5, 1.5)
social["high"] = fuzz.gaussmf(social.universe, 8, 1.5)

stress["low"] = fuzz.trimf(stress.universe, [0, 0, 5])
stress["moderate"] = fuzz.trimf(stress.universe, [3, 5, 7])
stress["high"] = fuzz.trimf(stress.universe, [5, 10, 10])

for rec in [sleep_rec, work_rec, activity_rec, social_rec]:
    rec["low"] = fuzz.trimf(rec.universe, [0, 0, 5])
    rec["medium"] = fuzz.trimf(rec.universe, [3, 5, 7])
    rec["high"] = fuzz.trimf(rec.universe, [5, 10, 10])

# -------------------------------------------------
# RULES
# -------------------------------------------------
rules = [
    ctrl.Rule(sleep["low"] & workload["high"], stress["high"]),
    ctrl.Rule(sleep["high"] & workload["low"], stress["low"]),
    ctrl.Rule(screen["high"] & physical["low"], stress["high"]),
    ctrl.Rule(workload["medium"] & sleep["medium"], stress["moderate"]),
    ctrl.Rule(social["high"] & workload["high"], stress["moderate"]),
    ctrl.Rule(physical["high"] & sleep["high"], stress["low"]),
    ctrl.Rule(screen["medium"] & workload["high"], stress["high"]),
    ctrl.Rule(social["low"] & workload["high"], stress["high"]),
    ctrl.Rule(physical["low"] & sleep["low"], stress["high"]),
    ctrl.Rule(screen["low"] & workload["low"], stress["low"]),
    ctrl.Rule(sleep["low"] & physical["low"] & social["low"], stress["high"])
]

rec_rules = [
    ctrl.Rule(sleep["low"], sleep_rec["high"]),
    ctrl.Rule(sleep["medium"], sleep_rec["medium"]),
    ctrl.Rule(sleep["high"], sleep_rec["low"]),

    ctrl.Rule(workload["high"], work_rec["high"]),
    ctrl.Rule(workload["medium"], work_rec["medium"]),
    ctrl.Rule(workload["low"], work_rec["low"]),

    ctrl.Rule(physical["low"], activity_rec["high"]),
    ctrl.Rule(physical["medium"], activity_rec["medium"]),
    ctrl.Rule(physical["high"], activity_rec["low"]),

    ctrl.Rule(social["low"], social_rec["high"]),
    ctrl.Rule(social["medium"], social_rec["medium"]),
    ctrl.Rule(social["high"], social_rec["low"]),
]

stress_ctrl = ctrl.ControlSystem(rules)
rec_ctrl = ctrl.ControlSystem(rec_rules)

# -------------------------------------------------
# BUTTON
# -------------------------------------------------
if st.button("🔍 Check Stress Level", use_container_width=True):

    try:
        sim = ctrl.ControlSystemSimulation(stress_ctrl)

        sim.input["sleep"] = sleep_val
        sim.input["workload"] = workload_val
        sim.input["screen"] = screen_val
        sim.input["physical"] = physical_val
        sim.input["social"] = social_val

        sim.compute()
        score = sim.output["stress"]

        if score < 4:
            level = "Low"
        elif score < 7:
            level = "Moderate"
        else:
            level = "High"

        # Inputs
        st.markdown("---")
        st.subheader("📥 Your Inputs")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"😴 Sleep: {sleep_val}")
            st.write(f"📚 Workload: {workload_val}")
            st.write(f"📱 Screen Time: {screen_val}")

        with col2:
            st.write(f"🏃 Physical Activity: {physical_val}")
            st.write(f"👥 Social Interaction: {social_val}")

        # Result
        st.markdown("---")
        st.subheader("📊 Result")
        st.progress(score / 10)
        st.metric("Stress Score", f"{score:.2f} / 10")

        if level == "Low":
            st.success(f"Stress Level: {level}")
        elif level == "Moderate":
            st.warning(f"Stress Level: {level}")
        else:
            st.error(f"Stress Level: {level}")

        # Fuzzy Recommendations
        st.subheader("💡 Fuzzy Recommendations")

        try:
            rec_sim = ctrl.ControlSystemSimulation(rec_ctrl)

            rec_sim.input["sleep"] = sleep_val
            rec_sim.input["workload"] = workload_val
            rec_sim.input["physical"] = physical_val
            rec_sim.input["social"] = social_val

            rec_sim.compute()
            shown = False

            if rec_sim.output["sleep_rec"] > 6:
                st.info("😴 High Priority: Improve sleep quality and schedule.")
                shown = True

            if rec_sim.output["work_rec"] > 6:
                st.info("📚 High Priority: Reduce workload and manage tasks.")
                shown = True

            if rec_sim.output["activity_rec"] > 6:
                st.info("🏃 High Priority: Increase physical activity.")
                shown = True

            if rec_sim.output["social_rec"] > 6:
                st.info("👥 High Priority: Improve social interaction and support.")
                shown = True

            if not shown:
                st.success("✅ Good balance detected. Maintain your healthy lifestyle.")

        except Exception:
            st.warning("Recommendations unavailable for this input.")

        # Graph
        st.subheader("📈 Stress Graph Explained")
        st.write("""
This graph compares your stress score with three zones:

🟢 Low Stress → Healthy and balanced  
🟡 Moderate Stress → Manage carefully  
🔴 High Stress → Warning zone
        """)

        st.write(
            f"📍 Your current stress score is **{score:.2f}**, "
            f"which falls under **{level} Stress**."
        )

        stress.view(sim=sim)
        st.pyplot(plt.gcf())

        if level == "Low":
            st.success("Interpretation: Your routine looks healthy and balanced.")
        elif level == "Moderate":
            st.warning("Interpretation: Your stress is manageable, but improvement is recommended.")
        else:
            st.error("Interpretation: Your stress is high. Consider lifestyle changes immediately.")

    except Exception:
        st.error("Unable to calculate stress for this input combination. Please adjust inputs and try again.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption("Developed using Fuzzy Logic + Streamlit")