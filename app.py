from flask import Flask, request, render_template
import pickle  # Changed from joblib to pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model using pickle instead of joblib
try:
    model = pickle.load(open('credit_risk_gradient_boosting.pkl', 'rb'))
    print("Model loaded successfully!")
except:
    print("Error: Model file not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user input from the form
    age = int(request.form['age'])
    sex = request.form['sex']
    job = int(request.form['job'])
    housing = request.form['housing']
    saving_accounts = request.form['saving_accounts']
    checking_account = request.form['checking_account']
    credit_amount = float(request.form['credit_amount'])
    duration = int(request.form['duration'])
    purpose = request.form['purpose']
    
    # Create a dataframe with the user input
    user_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    })
    
    # Handle NA values - replace empty strings with 'unknown'
    for col in ['Saving accounts', 'Checking account']:
        user_data[col] = user_data[col].replace('', 'unknown')
    
    # Feature engineering - create 'Credit per month' feature
    user_data['Credit per month'] = user_data['Credit amount'] / user_data['Duration']
    
    # Convert categorical variables to category dtype for consistency with training data
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        user_data[col] = user_data[col].astype('category')
    
    # Make prediction
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)[:, 1]  # Probability of good credit
    
    # Determine result based on the encoding in the dataset (good=1, bad=0)
    if prediction[0] == 1:
        risk_status = "Good Credit Risk"
        risk_class = "success"
    else:
        risk_status = "Bad Credit Risk"
        risk_class = "danger"
    
    confidence = probability[0] if prediction[0] == 1 else 1 - probability[0]
    
    return render_template('result.html', 
                          prediction=risk_status,
                          probability=round(confidence * 100, 2),
                          risk_class=risk_class,
                          user_data=user_data)

# Add a new route for batch prediction from CSV
@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('batch.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('batch.html', error="No file selected")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Handle missing values
            for col in ['Saving accounts', 'Checking account']:
                if col in df.columns:
                    df[col] = df[col].fillna('unknown')
            
            # Feature engineering
            df['Credit per month'] = df['Credit amount'] / df['Duration']
            
            # Convert categorical variables to category dtype
            categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            # Make predictions
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]
            
            # Add predictions to dataframe
            df['Prediction'] = ['Good Credit Risk' if p == 1 else 'Bad Credit Risk' for p in predictions]
            df['Probability'] = probabilities
            
            # Count results
            good_count = sum(predictions == 1)
            bad_count = sum(predictions == 0)
            
            return render_template('batch_results.html', 
                                  good_count=good_count,
                                  bad_count=bad_count,
                                  total=len(predictions),
                                  table=df.to_html(classes='table table-striped'))
            
        except Exception as e:
            return render_template('batch.html', error=str(e))
    
    return render_template('batch.html')

if __name__ == '__main__':
    app.run(debug=True)

