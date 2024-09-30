from flask import Flask, render_template, request, jsonify
import pandas as pd
from ml_model import ml_model, Preprocessiong
import easyocr


app = Flask(__name__)

report_dataset = pd.read_csv('diabities_dataset.csv')
Preprocessiong(report_dataset)
model = ml_model(report_dataset)

def extract_text_from_image(image_path):
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path)
        extracted_text = ' '.join([text[1] for text in result])
        return extracted_text
    except Exception as e:
        return {'error': str(e)}

def extract_patient_data(text):
    extracted_data = {}
    try:
        extracted_data['Pregnancies'] = extract_patient_Pregnancies(text)
        extracted_data['Glucose'] = extract_patient_Glucose(text)
        extracted_data['BloodPressure'] = extract_patient_BloodPressure(text)
        extracted_data['SkinThickness'] = extract_patient_SkinThickness(text)
        extracted_data['BMI'] = extract_patient_BMI(text)
        extracted_data['DiabetesPedigreeFunction'] = extract_patient_DiabetesPedigreeFunction(text)
        extracted_data['Age'] = extract_patient_age(text)
        return extracted_data
    except Exception as e:
        return {'error': str(e)}

def extract_patient_age(text):  
    try:
        age_index = text.find("Age")
        if age_index != -1:
            age_info = text[age_index + 3:].split()
            if age_info:
                return int(age_info[0])
    except:
        return 35

def extract_patient_Glucose(text):
    try:
        glucose_index = text.find("Glucose")
        if glucose_index != -1:
            glucose_info = text[glucose_index + 7:].split()
            if glucose_info:
                return int(glucose_info[0])
    except:
        return 0

def extract_patient_Pregnancies(text):
    try:
        Pregnancies_index = text.find('Pregnancies')
        if Pregnancies_index != -1:
            Pregnancies_info = text[Pregnancies_index + 11:].split()
            if Pregnancies_info:
                return float(Pregnancies_info[0])
    except :
        return 0

def extract_patient_BloodPressure(text):
    try:
        BloodPressure_index = text.find('Pressure')
        if BloodPressure_index != -1:
            BloodPressure_info = text[BloodPressure_index + 8:].split()
            if BloodPressure_info:
                return int(BloodPressure_info[0])
    except:
        return 0

def extract_patient_SkinThickness(text):
    try:
        Pregnancies_SkinThickness = text.find('Thickness')
        if Pregnancies_SkinThickness != -1:
            SkinThickness_info = text[Pregnancies_SkinThickness + 9:].split()
            if SkinThickness_info:
                return int(SkinThickness_info[0])
    except:
         return 0

def extract_patient_BMI(text):
    try:
        BMI_index = text.find('BMI')
        if BMI_index != -1:
            BMI_info = text[BMI_index + 3:].split()
            if BMI_info:
                return float(BMI_info[0])
    except:
        return 18.5

def extract_patient_DiabetesPedigreeFunction(text):
    try:
        DiabetesPedigreeFunction_index = text.find('Function')
        if DiabetesPedigreeFunction_index != -1:
            DiabetesPedigreeFunction_info = text[DiabetesPedigreeFunction_index + 9:].split()
            if DiabetesPedigreeFunction_info:
                return float(DiabetesPedigreeFunction_info[0])
    except :
        return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'report' not in request.files:
        return jsonify({'error': 'No file part'})
    
    report_file = request.files['report']
    if report_file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_path = 'temp_image.jpg'
    report_file.save(image_path)

    extracted_text = extract_text_from_image(image_path)
    if 'error' in extracted_text:
        return jsonify(extracted_text)

    extracted_data = extract_patient_data(extracted_text)
    if 'error' in extracted_data:
        return jsonify(extracted_data)

    prediction_value = model.predict([[extracted_data['Pregnancies'], extracted_data['Glucose'], extracted_data['BloodPressure'], extracted_data['SkinThickness'], 0, extracted_data['BMI'], extracted_data['DiabetesPedigreeFunction'], extracted_data['Age']]])

    if prediction_value == 1:
        prediction = "Patient Has Diabetes"
    else:
        prediction = "Patient Does Not have Diabetes"

    return jsonify({'prediction': prediction})

@app.route('/check_report')
def check_report():
    return render_template('report_reader.html')

@app.route('/services')
def services():
    return render_template('service.html')
@app.route('/contact')
def contact():
    return render_template('contact_us.html')

@app.route('/about')
def about():
    return render_template('about_us.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/home')
def home():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
