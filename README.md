HairBot is an AI-powered web application that predicts a user’s Norwood stage of hair loss from a selfie image and provides personalized treatment recommendations. It also includes a hair loss progression simulator that shows how the user’s hair might look in 5–6 years if left untreated, using a CycleGAN-based image-to-image translation model.

🚀 Features
✅ Norwood Stage Detection (1–7) using a TensorFlow MobileNetV2 classifier
✅ Questionnaire-based Hair Loss Diagnosis with treatment recommendations
✅ Downloadable Diagnosis Report
✅ Hair Loss Progression Simulation using CycleGAN
✅ Streamlit-based Web Interface with an easy-to-use UI

🏗️ Project Structure
HairLoss_Predictor/
│── unified_dataset/               # Original dataset (images + CSV)
│── balanced_dataset/              # Final balanced dataset (after augmentation)
│── norwood_mobilenet_model.h5     # Trained Norwood classifier model
│── cyclegan_data/                 # CycleGAN training data (trainA/trainB/testA/testB)
│── CycleGAN/                      # CycleGAN training notebooks
│── Hairbot.py                     # Streamlit app
│── detection.ipynb                # Model training notebook
│── README.md                      # This file

⚡ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/HairBot.git
cd HairBot
2️⃣ Create a virtual environment & install dependencies
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
pip install -r requirements.txt
3️⃣ Install additional libraries
pip install tensorflow torch torchvision streamlit pillow opencv-python pandas numpy matplotlib seaborn tqdm

📂 Dataset
Norwood Stage Classifier Dataset: A balanced dataset with 20 images per stage (1–7), augmented using Keras’ ImageDataGenerator.
CycleGAN Dataset: Images are divided into trainA, trainB, testA, testB for each stage-to-stage mapping (e.g., Stage 1 → 2).

🔥 Model Details
✅ Norwood Classifier (TensorFlow)
Base Model: MobileNetV2 (transfer learning)

Input Size: 224×224×3

Output: 7 classes (Norwood stages 1–7)

✅ Hair Loss Progression Simulator (CycleGAN)
Mapping:

Stage 1 → 2

Stage 2 → 3

Stage 3 → 5

Stage 4 → 6

Stage 5/6 → 7

🎯 How to Use
🔹 Run the Web App
bash
Copy code
streamlit run Hairbot.py
🔹 Upload a Selfie Image
The app will predict your Norwood stage.
You will get personalized treatment recommendations.
You can download a diagnosis report.

🔹 Simulate Future Hair Loss
A CycleGAN-based model predicts how your hair might look after 5–6 years without treatment.

📊 Example Workflow
1️⃣ User uploads an image
2️⃣ Model predicts Norwood stage (e.g., Stage 2)
3️⃣ App recommends treatment & tests
4️⃣ CycleGAN generates Stage 4 (future hair loss) image
5️⃣ User downloads a report

🛠️ Technologies Used
Category	Technology
Frontend	Streamlit
Model (Classifier)	TensorFlow (MobileNetV2)
Model (CycleGAN)	PyTorch
Image Processing	OpenCV, Pillow
Data Handling	Pandas, NumPy

🔮 Future Enhancements
✅ Replace CycleGAN with Pix2Pix for paired image training
✅ Add real-time webcam prediction
✅ Deploy app on Streamlit Cloud or Render
✅ Add treatment progress tracking (before/after comparison)






