from model import predict_class, import_model
import os
from flask import Flask, render_template, request

# Create flask app
app = Flask(__name__)

# Create folder to upload
app.config['UPLOAD_FOLDER'] = os.path.basename('Uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

# Import model
model = import_model('iceberg_model.h5')


# Root endpoint
@app.route('/')
def upload_file():
    return render_template('index.html')


# Upload endpoint
@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        current_dir = os.getcwd()
        img = request.files['file']
        file_path = os.path.join(current_dir, app.config['UPLOAD_FOLDER'], img.filename)
        img.save(file_path)

        prediction = predict_class(model, file_path)
        if prediction[0][0] < 0.5:
            prediction_message = 'This is New Jersey Devils player with ' + str((1 - prediction[0][0]) * 100) + '% probability!'
        else:
            prediction_message = 'This is Carolina Hurricanes player with ' + str(prediction[0][0] * 100) + '% probability!'

        return render_template('index.html', prediction_message=prediction_message)


if __name__ == "__main__":
    app.run()
