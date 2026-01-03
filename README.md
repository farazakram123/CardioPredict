# CardioPredict - Heart Attack Risk Prediction 🫀

A machine learning-powered web application that predicts the risk of heart attack based on various cardiovascular health indicators. Built with Flask and scikit-learn, this application provides an intuitive interface for healthcare professionals and individuals to assess heart disease risk.

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Input Features](#input-features)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Real-time Prediction**: Instant heart attack risk assessment based on user input
- **Interactive Web Interface**: User-friendly form with clear visual feedback
- **Probability Scoring**: Provides risk percentage and confidence levels
- **Machine Learning Backend**: Trained classification model with data preprocessing
- **Responsive Design**: Modern UI with gradient backgrounds and smooth animations
- **RESTful API**: JSON-based API for predictions

## 🎯 Demo

The application takes 11 cardiovascular health parameters as input and predicts:
- ✓ No Risk of Heart Attack
- ⚠️ Risk of Heart Attack Detected

Along with:
- Risk percentage
- Prediction confidence level

## 📊 Dataset

The project uses the **Heart Disease dataset** (`heart.csv`) containing 920 patient records with the following attributes:

- **Age**: Patient's age in years
- **Sex**: M (Male) or F (Female)
- **ChestPainType**: ATA (Atypical Angina), NAP (Non-Anginal Pain), ASY (Asymptomatic), TA (Typical Angina)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dl)
- **FastingBS**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **RestingECG**: Resting electrocardiogram results (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y = Yes; N = No)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment (Up, Flat, Down)
- **HeartDisease**: Target variable (1 = heart disease; 0 = normal)

## 🛠️ Technologies Used

### Backend
- **Python 3.x**
- **Flask**: Web framework for the application
- **scikit-learn**: Machine learning library for model training and prediction
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Frontend
- **HTML5**
- **CSS3**: Custom styling with gradients and animations
- **JavaScript**: Form handling and AJAX requests

### Data Science
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **Jupyter Notebook**: Exploratory data analysis and model development

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HeartDisease
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if model files don't exist)
   - Open `prediction.ipynb` in Jupyter Notebook
   - Run all cells to train the model and generate:
     - `cardio_prediction_model.pkl`
     - `scaler.pkl`
   
   Or run:
   ```bash
   jupyter notebook prediction.ipynb
   ```

## 💻 Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your web browser
   - Navigate to `http://127.0.0.1:5000/`

3. **Make a prediction**
   - Fill in all the required health parameters
   - Click "Predict Risk"
   - View the prediction result with risk percentage

### Running in Debug Mode

The application runs in debug mode by default. To disable debug mode, modify `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=False)
```

## 🧠 Model Training

The machine learning model is trained using the following pipeline:

1. **Data Preprocessing**
   - One-hot encoding for categorical variables
   - Feature scaling using StandardScaler
   - Train-test split (typically 80-20)

2. **Model Selection**
   - Classification algorithm (likely Logistic Regression, Random Forest, or similar)
   - Hyperparameter tuning
   - Cross-validation

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Score

4. **Model Persistence**
   - Trained model saved as `cardio_prediction_model.pkl`
   - Scaler saved as `scaler.pkl`

All training code is available in `prediction.ipynb`.

## 📁 Project Structure

```
HeartDisease/
│
├── app.py                          # Flask application (main server)
├── heart.csv                       # Dataset for training
├── prediction.ipynb                # Jupyter notebook for model training
├── requirements.txt                # Python dependencies
├── cardio_prediction_model.pkl     # Trained ML model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── README.md                       # Project documentation
│
└── templates/
    └── index.html                  # Frontend HTML interface
```

## 📝 Input Features

### Required Parameters

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| Age | Integer | Patient's age in years | 40, 55, 60 |
| Sex | Categorical | M or F | M, F |
| ChestPainType | Categorical | Type of chest pain | ATA, NAP, ASY, TA |
| RestingBP | Integer | Resting blood pressure (mm Hg) | 120, 140, 160 |
| Cholesterol | Integer | Serum cholesterol (mg/dl) | 200, 250, 300 |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dl | 0, 1 |
| RestingECG | Categorical | Resting ECG results | Normal, ST, LVH |
| MaxHR | Integer | Maximum heart rate | 150, 170, 190 |
| ExerciseAngina | Categorical | Exercise-induced angina | Y, N |
| Oldpeak | Float | ST depression | 0.0, 1.5, 2.5 |
| ST_Slope | Categorical | Slope of peak exercise ST | Up, Flat, Down |

## 🔌 API Endpoints

### GET `/`
Returns the main HTML interface.

**Response**: HTML page

### POST `/predict`
Makes a heart attack risk prediction based on input parameters.

**Request Format**: Form data with all 11 parameters

**Response Format**: JSON
```json
{
  "result": "⚠️ Risk of Heart Attack Detected",
  "result_class": "risk",
  "risk_percentage": "75.43%",
  "confidence": "75.43%"
}
```

**Error Response**: JSON
```json
{
  "error": "Error message description"
}
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- Add more visualization features
- Implement model comparison (try different algorithms)
- Add user authentication and history tracking
- Create a mobile-responsive design
- Add more comprehensive error handling
- Implement unit tests
- Add Docker support

## 📄 License

This project is available for educational and research purposes. Please ensure compliance with data privacy regulations when using health-related data.

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 👨‍💻 Author

MOHD FARAZ AKRAM

---

**Note**: Make sure to train the model using `prediction.ipynb` before running the Flask application for the first time.
