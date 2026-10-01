import streamlit as st
import pandas as pd
import joblib

# ── 1. LOAD MODEL AND FEATURE COLUMNS ────────────────────────────────────────
model = joblib.load("models/random_forest.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ── 2. PAGE SETUP ─────────────────────────────────────────────────────────────
st.title("E-Commerce Customer Churn Prediction")
st.write("Fill in the customer details below and click **Predict** to check if the customer is likely to churn.")

# ── 3. INPUT FORM ─────────────────────────────────────────────────────────────
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    Tenure                      = st.number_input("Tenure (months)",                   min_value=0,   max_value=60,  value=10)
    WarehouseToHome             = st.number_input("Warehouse To Home (km)",             min_value=0,   max_value=130, value=15)
    HourSpendOnApp              = st.number_input("Hour Spend On App",                  min_value=0,   max_value=5,   value=3)
    NumberOfDeviceRegistered    = st.number_input("Number Of Device Registered",        min_value=1,   max_value=6,   value=3)
    SatisfactionScore           = st.number_input("Satisfaction Score (1–5)",           min_value=1,   max_value=5,   value=3)
    NumberOfAddress             = st.number_input("Number Of Address",                  min_value=1,   max_value=22,  value=3)
    OrderAmountHikeFromlastYear = st.number_input("Order Amount Hike From Last Year (%)", min_value=11, max_value=26, value=15)
    CouponUsed                  = st.number_input("Coupon Used",                        min_value=0,   max_value=16,  value=1)
    OrderCount                  = st.number_input("Order Count",                        min_value=1,   max_value=16,  value=3)
    DaySinceLastOrder           = st.number_input("Day Since Last Order",               min_value=0,   max_value=46,  value=5)

with col2:
    CashbackAmount      = st.number_input("Cashback Amount",            min_value=0,   max_value=325, value=150)
    CityTier            = st.selectbox("City Tier",                     [1, 2, 3])
    Complain            = st.selectbox("Complain (0 = No, 1 = Yes)",    [0, 1])
    PreferredLoginDevice = st.selectbox("Preferred Login Device",       ["Mobile Phone", "Computer", "Phone"])
    PreferredPaymentMode = st.selectbox("Preferred Payment Mode",       ["Debit Card", "UPI", "Credit Card", "Cash on Delivery", "E wallet", "COD", "CC"])
    Gender              = st.selectbox("Gender",                        ["Male", "Female"])
    PreferedOrderCat    = st.selectbox("Preferred Order Category",      ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"])
    MaritalStatus       = st.selectbox("Marital Status",                ["Single", "Married", "Divorced"])

# ── 4. PREDICT BUTTON ─────────────────────────────────────────────────────────
if st.button("Predict"):

    # Build a single-row DataFrame with raw input (same column names as original dataset)
    input_data = pd.DataFrame([{
        "Tenure":                       Tenure,
        "CityTier":                     CityTier,
        "WarehouseToHome":              WarehouseToHome,
        "HourSpendOnApp":               HourSpendOnApp,
        "NumberOfDeviceRegistered":     NumberOfDeviceRegistered,
        "SatisfactionScore":            SatisfactionScore,
        "NumberOfAddress":              NumberOfAddress,
        "Complain":                     Complain,
        "OrderAmountHikeFromlastYear":  OrderAmountHikeFromlastYear,
        "CouponUsed":                   CouponUsed,
        "OrderCount":                   OrderCount,
        "DaySinceLastOrder":            DaySinceLastOrder,
        "CashbackAmount":               CashbackAmount,
        "PreferredLoginDevice":         PreferredLoginDevice,
        "PreferredPaymentMode":         PreferredPaymentMode,
        "Gender":                       Gender,
        "PreferedOrderCat":             PreferedOrderCat,
        "MaritalStatus":                MaritalStatus,
    }])

    # Apply same One-Hot Encoding as train.py (pd.get_dummies with drop_first=True)
    input_encoded = pd.get_dummies(input_data, drop_first=True)

    # Align columns to exactly match training feature columns
    # Add any missing columns as 0, remove any extra columns
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # ── 5. MAKE PREDICTION ────────────────────────────────────────────────────
    prediction  = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]  # probability of churn

    # ── 6. DISPLAY RESULTS ────────────────────────────────────────────────────
    st.header("Prediction Result")

    if prediction == 1:
        st.error("Churn Prediction: YES — This customer is likely to churn.")
    else:
        st.success("Churn Prediction: NO — This customer is likely to stay.")

    st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")

    # Risk Level based on probability
    if probability < 0.3:
        st.info("Risk Level: 🟢 Low")
    elif probability < 0.6:
        st.warning("Risk Level: 🟡 Medium")
    else:
        st.error("Risk Level: 🔴 High")
