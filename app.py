from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    model = None
    print("Error: 'model.pkl' file not found.")
except pickle.UnpicklingError:
    model = None
    print("Error: Unable to load the 'model.pkl' file.")


# @app.route('/', methods=['GET', 'POST'])
# def home():
#     """Render the input form."""
#     return render_template('work.html')

@app.route('/')
def home():
    """Render the input form and handle the 'View Scholarship' click."""
    if request.method == 'POST':
        # This handles the form submission
        try:
            # Retrieve user inputs from the form
            age = int(request.form['age'])
            income = float(request.form['income'])
            marks = float(request.form['marks'])
            siblings = int(request.form['siblings'])
            location=int(request.form['location'])
            class_level = int(request.form['class_level'])
            school_type = int(request.form['school_type'])  # 0 or 1 (encoded)
            gender = int(request.form['gender'])  # 0 or 1 (encoded)

            # Combine inputs into a feature array
            input_features = np.array([[age, income, marks, siblings,location, class_level, school_type, gender]])

            # Predict the result
            prediction = model.predict(input_features)[0]

            # If likely to drop out, display the message with a link to the scholarship page
            if prediction == 1:
                return render_template(
                    'work.html',
                    prediction_text="Prediction: Based on the above input data, the student is likely to Dropout.",
                    show_scholar_button=True,
                    warning_message=""
                )

            # Otherwise, display "Not Likely to Dropout"
            result = 'Student will not Dropout'
            return render_template('work.html', prediction_text=f"Prediction: {result}", warning_message="")

        except ValueError as ve:
            return render_template('work.html', prediction_text=f"Error: Invalid input - {ve}", warning_message="")

        except Exception as e:
            return render_template('work.html', prediction_text=f"Error: {e}", warning_message="")

    else:
        # This handles when the page is loaded but no form is submitted
        return render_template('work.html', warning_message="")



@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction logic."""
    try:
        if model is None:
            return render_template('work.html', prediction_text="Error: Model not loaded.")
        # Retrieve user inputs from the form

        age = int(request.form['age'])
        income = float(request.form['income'])
        marks = float(request.form['marks'])
        siblings = int(request.form['siblings'])
        location = int(request.form['location'])
        class_level = int(request.form['class_level'])
        school_type = int(request.form['school_type'])
        gender = int(request.form['gender'])

        if age < 6 or age > 25:
            return render_template('work.html', prediction_text="Error: Age must be between 6 and 25.")

        if income < 60000 or income > 400000:
            return render_template('work.html', prediction_text="Error: Income must be between 60000 and 300000.")

        if marks < 0 or marks > 100:
            return render_template('work.html', prediction_text="Error: Marks must be between 0 and 100.")

        if class_level < 3 or class_level >12:
            return render_template('work.html', prediction_text="Error: Class level must be between 3 and 12.")

        # Enforce range restriction on siblings
        if siblings < 0 or siblings > 5:
            return render_template('work.html', prediction_text="Error: Siblings must be between 0 and 5.")


        # Combine inputs into a feature array
        input_features = np.array([[age, income, marks, siblings, location,class_level, school_type, gender]])

        # Predict the result
        prediction = model.predict(input_features)[0]

        # If likely to drop out, display the message with a link to the scholarship page
        if prediction == 1:
            return render_template(
                'work.html',
                prediction_text="Prediction: Based on the above input data student is likely to Dropout.",
                show_scholar_button=True
            )

        # Otherwise, display "Not Likely to Dropout"
        result = 'Student will not Dropout'
        return render_template('work.html', prediction_text=f"Prediction: {result}")

    except ValueError as ve:
        return render_template('work.html', prediction_text=f"Error: Invalid input - {ve}")

    except Exception as e:
        return render_template('work.html', prediction_text=f"Error: {e}")


# New route for scholarship links page
@app.route('/scholar')
def scholar():
    """Render the scholarship links page."""
    return render_template('scholar.html')


if __name__ == "__main__":
    app.run(host='localhost', port=5003)

    #app.run(debug=True)
