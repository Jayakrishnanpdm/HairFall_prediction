import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from datetime import datetime

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stSelectbox, .stTextInput, .stRadio, .stMultiselect {
        background-color: #2c2c2c;
        border-radius: 8px;
        padding: 10px;
        color: #ffffff;
    }
    .stFileUploader {
        background-color: #2c2c2c;
        border-radius: 8px;
        padding: 10px;
    }
    .stExpander {
        background-color: #252525;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid #4CAF50;
    }
    .stExpander .st-expander {
        background-color: #2c2c2c;
        color: #ffffff;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4CAF50;
        font-weight: bold;
    }
    .diagnosis-box {
        background-color: #333333;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .warning-box {
        background-color: #ff4444;
        padding: 10px;
        border-radius: 8px;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .stCaption {
        color: #aaaaaa;
        text-align: center;
        margin-top: 20px;
    }
    .stTextInput input, .stSelectbox select, .stRadio label, .stMultiselect div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CSS styling and headers ---
st.title("✨ HairBot - Your Ultimate Hair Health Assistant")
st.markdown("An AI-powered tool to diagnose hair loss causes and recommend personalized treatments with precision.", unsafe_allow_html=True)

# --- Load Norwood model ---
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

norwood_model = load_model("norwood_mobilenet_model.h5")  # Adjust path if needed

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_norwood_stage(img):
    processed = preprocess_image(img)
    pred = norwood_model.predict(processed)
    return np.argmax(pred) + 1

# Progress bar setup
sections = ["Basic Information", "Hair Loss History", "Medical History", "Hormonal/Reproductive", "Lifestyle & Diet", "Hair Care Routine", "Family History"]
completed_sections = 0
total_sections = len(sections)

# Symptom selection
symptom = st.selectbox(
    "Select a hair issue: 🧬",
    ["", "Dandruff", "Dry Hair", "Hair Fall", "Oily Scalp", "Itchy Scalp"],
    help="Choose the primary hair issue you're experiencing."
)

if symptom == "Hair Fall":
    # Basic Information
    with st.expander("👤 Basic Information", expanded=True):
        name = st.text_input("What is your name?", placeholder="Enter your name")
        age = st.text_input("What is your age?", placeholder="Enter your age (e.g., 30)")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Select your gender")
        if name and age and gender:
            completed_sections += 1
        if not age.isdigit() and age:
            st.markdown("<div class='warning-box'>⚠️ Please enter a valid age (numbers only).</div>", unsafe_allow_html=True)

    # Hair Loss History
    with st.expander("🕒 Hair Loss History"):
        onset = st.selectbox("When did you first notice hair fall?", ["< 1 month ago", "1–3 months ago", "> 3 months ago"])
        speed = st.selectbox("Was the hair loss sudden or gradual?", ["Sudden", "Gradual"])
        pattern = st.selectbox("How is the hair falling?", ["All over scalp", "In patches", "Mainly hairline/crown"])
        if onset and speed and pattern:
            completed_sections += 1

    # Medical History
    with st.expander("🧬 Medical History"):
        conditions = st.multiselect("Do you have any of these conditions?", ["Thyroid problems", "Anemia", "Autoimmune diseases", "PCOS", "None"])
        meds = st.radio("Are you taking any medications regularly?", ["Yes", "No"])
        if conditions and meds:
            completed_sections += 1
        if "None" in conditions and len(conditions) > 1:
            st.markdown("<div class='warning-box'>⚠️ 'None' cannot be selected with other conditions.</div>", unsafe_allow_html=True)

    # Hormonal/Reproductive
    with st.expander("♀️ Hormonal/Reproductive (if applicable)"):
        hormonal_issues = st.multiselect("Have you experienced any of these?", ["Pregnancy/childbirth", "Irregular periods", "Menopausal symptoms", "Excess facial hair/acne", "None"])
        if hormonal_issues:
            completed_sections += 1
        if gender != "Female" and hormonal_issues and "None" not in hormonal_issues:
            st.markdown("<div class='warning-box'>⚠️ Hormonal issues are typically relevant for female patients. Please confirm.</div>", unsafe_allow_html=True)

    # Lifestyle & Diet
    with st.expander("🍽️ Lifestyle & Diet"):
        diet_type = st.selectbox("What is your diet type?", ["Non-vegetarian", "Vegetarian", "Vegan"], help="Select your dietary preference")
        diet_change = st.radio("Recent diet change or weight loss?", ["Yes", "No"])
        stress = st.selectbox("Rate your stress levels recently", ["Low", "Moderate", "High"])
        if diet_type and diet_change and stress:
            completed_sections += 1
        if stress == "High":
            st.markdown("<div class='warning-box'>⚠️ High stress can contribute to hair loss. Consider stress management techniques.</div>", unsafe_allow_html=True)
        if diet_type in ["Vegetarian", "Vegan"] and diet_change == "Yes":
            st.markdown("<div class='warning-box'>⚠️ Vegetarian/vegan diets with recent changes may lead to nutritional deficiencies. Ensure adequate protein, iron, and B12 intake.</div>", unsafe_allow_html=True)

    # Hair Care Routine
    with st.expander("🧴 Hair Care Routine"):
        tight_styles = st.radio("Do you wear tight hairstyles frequently?", ["Yes", "No"])
        heat_tools = st.selectbox("How often do you use heat or chemical treatments?", ["Frequently", "Occasionally", "Never"])
        scalp_issues = st.radio("Any recent scalp irritation, flaking, or itching?", ["Yes", "No"])
        water_quality = st.radio("Do you use hard water or chlorinated water (e.g., pool water, untreated tap water)?", ["Yes", "No"])
        if tight_styles and heat_tools and scalp_issues and water_quality:
            completed_sections += 1
        if tight_styles == "Yes":
            st.markdown("<div class='warning-box'>⚠️ Tight hairstyles may cause traction alopecia. Consider looser styles.</div>", unsafe_allow_html=True)
        if water_quality == "Yes":
            st.markdown("<div class='warning-box'>⚠️ Hard or chlorinated water may cause hair breakage or scalp irritation. Consider using a shower filter or clarifying shampoo.</div>", unsafe_allow_html=True)

    # Family History
    with st.expander("👪 Family History"):
        family_history = st.radio("Any family history of hair loss?", ["Yes", "No", "Not sure"])
        if family_history:
            completed_sections += 1

    # Progress bar
    progress = completed_sections / total_sections
    st.progress(progress)
    st.markdown(f"**Form Completion: {int(progress * 100)}%** ({completed_sections}/{total_sections} sections completed)", unsafe_allow_html=True)

    # Image Upload
    uploaded_file = st.file_uploader("Upload a clear photo of your hair/scalp", type=["jpg", "jpeg", "png"], help="Upload a high-quality image (max 5MB)")
    with st.expander("📸 Upload Hair Image for Norwood Stage Detection"):
        if uploaded_file:
            # Basic image validation
            img = Image.open(uploaded_file)
            if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
                st.markdown("<div class='warning-box'>⚠️ Image size exceeds 5MB. Please upload a smaller file.</div>", unsafe_allow_html=True)
            else:
                st.image(img, caption="Uploaded Hair/Scalp Image", use_column_width=True)
            
    # Norwood treatment dictionary
    norwood_treatment = {
        "1": {
            "treatment": "No treatment needed, but 5% minoxidil can be used as prevention.",
            "reversible": True
        },
        "2": {
            "treatment": "5% minoxidil + 0.1% topical finasteride (morning and night).",
            "reversible": True
        },
        "3": {
            "treatment": "10% minoxidil + oral or topical finasteride. Microneedling twice a week helps.",
            "reversible": True
        },
        "3V": {
            "treatment": "10% minoxidil + oral finasteride. Focus on crown treatment.",
            "reversible": True
        },
        "4": {
            "treatment": "High-strength minoxidil (10%) + oral finasteride. Microneedling recommended.",
            "reversible": True
        },
        "5": {
            "treatment": "Medical treatment might slow loss, but hair transplant is usually needed.",
            "reversible": False
        },
        "6": {
            "treatment": "Hair transplant is the most effective option. Meds can preserve donor hair.",
            "reversible": False
        },
        "7": {
            "treatment": "Only solution: hair transplant. Maintain donor area with oral finasteride.",
            "reversible": False
        }
    }

    def recommend_treatment(stage):
        stage_str = str(stage)
        if stage_str not in norwood_treatment:
            return "Unknown Norwood stage."

        data = norwood_treatment[stage_str]
        status = "🟢 **Reversible**" if data["reversible"] else "🔴 **Not Reversible** (Transplant Likely Needed)"

        return f"""
### 🧠 Detected Norwood Stage: **Stage {stage}**
**💊 Recommended Treatment:**  
{data['treatment']}

**{status}**
**Note**: This is a placeholder. Integrate an ML model for accurate Norwood stage detection.
"""

if uploaded_file and uploaded_file.size <= 5 * 1024 * 1024:
    # Reload the image freshly
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Hair/Scalp Image", use_column_width=True)

    # Predict and show stage
    stage = predict_norwood_stage(img)
    st.write(f"🔎 **Model Prediction: Stage {stage}**")  # Optional debug line
    st.markdown(recommend_treatment(stage), unsafe_allow_html=True)


    # Diagnosis Summary
    with st.expander("📋 Diagnosis Summary", expanded=True):
        diagnosis = ""
        tests = []
        if not name or not age or not gender:
            st.markdown("<div class='warning-box'>⚠️ Please complete the Basic Information section for a diagnosis.</div>", unsafe_allow_html=True)
        if completed_sections < total_sections:
            st.markdown("<div class='warning-box'>⚠️ Please complete all sections for a comprehensive diagnosis.</div>", unsafe_allow_html=True)
        else:
            if stress == "High" and diet_change == "Yes":
                diagnosis += "Your hair fall may be due to **telogen effluvium** triggered by stress or nutritional changes.\n"
                tests.append("Ferritin, Vitamin D, Zinc, Thyroid function (TSH, T3, T4)")
            if pattern == "In patches" and scalp_issues == "No":
                diagnosis += "Patchy hair loss without scalp irritation suggests **alopecia areata**. Consult a dermatologist for autoimmune evaluation.\n"
                tests.append("Autoimmune markers (ANA, anti-thyroid antibodies)")
            if "Thyroid problems" in conditions:
                diagnosis += "Thyroid disorders may contribute to hair loss. Ensure optimal TSH levels.\n"
                tests.append("Thyroid function tests (TSH, T3, T4)")
            if "PCOS" in conditions or hormonal_issues:
                diagnosis += "Hormonal imbalances (e.g., PCOS, pregnancy) may be contributing. Consult an endocrinologist.\n"
                tests.append("Hormone panel (testosterone, DHEA-S, LH/FSH, prolactin)")
            if tight_styles == "Yes":
                diagnosis += "Frequent tight hairstyles may cause **traction alopecia**. Switch to looser styles to prevent further damage.\n"
                tests.append("Trichoscopy (if loss persists)")
            if family_history == "Yes" and pattern == "Mainly hairline/crown":
                diagnosis += "Your hair loss pattern and family history suggest **androgenetic alopecia**. Consider minoxidil or finasteride.\n"
                tests.append("Hormone panel (for women, if PCOS suspected)")
            if scalp_issues == "Yes":
                diagnosis += "Scalp irritation or flaking may indicate **seborrheic dermatitis** or **fungal infection**. Consult a dermatologist.\n"
                tests.append("Fungal culture or KOH test")
            if water_quality == "Yes":
                diagnosis += "Exposure to **hard or chlorinated water** may contribute to hair breakage or scalp irritation. Use a shower filter or clarifying shampoo to reduce mineral buildup.\n"
                tests.append("Water quality test (for mineral content or chlorine levels)")
            if diet_type in ["Vegetarian", "Vegan"] and diet_change == "Yes":
                diagnosis += f"Your {diet_type.lower()} diet with recent changes may lead to **nutritional deficiencies**. Ensure adequate protein, iron, and B12 intake.\n"
                tests.append("Ferritin, Vitamin B12, Serum protein")

            if diagnosis:
                st.markdown(f"<div class='diagnosis-box'>{diagnosis}\n**Note**: Always consult a dermatologist to confirm this diagnosis and discuss treatment options.</div>", unsafe_allow_html=True)
                if tests:
                    st.markdown("### 🩺 Recommended Tests")
                    st.write("- " + "\n- ".join(tests))
            else:
                st.info("No specific diagnosis pattern matched. Please provide more details or consult a dermatologist.")

        # Download Diagnosis Report
        if st.button("📥 Download Diagnosis Report"):
            report = f"""
# HairBot Diagnosis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Patient Information
- **Name**: {name or 'Not provided'}
- **Age**: {age or 'Not provided'}
- **Gender**: {gender or 'Not provided'}

## Hair Loss Details
- **Onset**: {onset or 'Not provided'}
- **Speed**: {speed or 'Not provided'}
- **Pattern**: {pattern or 'Not provided'}

## Medical and Lifestyle Factors
- **Conditions**: {', '.join(conditions) if conditions else 'Not provided'}
- **Medications**: {meds or 'Not provided'}
- **Hormonal Issues**: {', '.join(hormonal_issues) if hormonal_issues else 'Not provided'}
- **Diet Type**: {diet_type or 'Not provided'}
- **Diet Change/Weight Loss**: {diet_change or 'Not provided'}
- **Stress Level**: {stress or 'Not provided'}
- **Tight Hairstyles**: {tight_styles or 'Not provided'}
- **Heat/Chemical Treatments**: {heat_tools or 'Not provided'}
- **Scalp Issues**: {scalp_issues or 'Not provided'}
- **Water Quality (Hard/Chlorinated)**: {water_quality or 'Not provided'}
- **Family History**: {family_history or 'Not provided'}

## Diagnosis
{diagnosis or 'No specific diagnosis matched due to incomplete information.'}

## Recommended Tests
{', '.join(tests) if tests else 'None recommended due to incomplete information.'}

*Generated by HairBot. Consult a dermatologist for professional advice.*
"""
            b = io.BytesIO(report.encode())
            st.download_button(
                label="Download Report (TXT)",
                data=b,
                file_name=f"hairbot_report_{name or 'user'}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            st.success("Report downloaded successfully!")

elif symptom:
    with st.expander("🔍 Suggested Action", expanded=True):
        if symptom == "Dandruff":
            st.write("For **Dandruff**, use shampoos with ketoconazole, zinc pyrithione, or salicylic acid. Avoid harsh shampoos and oily products.")
        elif symptom == "Dry Hair":
            st.write("For **Dry Hair**, apply hydrating masks, avoid frequent heat styling, and use leave-in conditioners with argan oil or shea butter.")
        elif symptom == "Oily Scalp":
            st.write("For **Oily Scalp**, wash regularly with a mild clarifying shampoo and avoid heavy conditioners on the scalp.")
        elif symptom == "Itchy Scalp":
            st.write("For **Itchy Scalp**, try anti-dandruff shampoos and avoid fragranced products. Tea tree oil shampoos can help.")
        else:
            st.write(f"For **{symptom}**, consult a dermatologist for personalized care.")

# Footer
st.markdown("---")
st.caption("🔬 Powered by HairBot | Created by Nitin and Jaykrishnan | Built with Streamlit")