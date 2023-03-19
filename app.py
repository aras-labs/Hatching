from datetime import datetime
import os
from flask import Flask, flash, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename

from main import convert

UPLOAD_FOLDER = 'images'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/outputs/<name>')
def download_file(name):
    full_path = os.path.join(app.config['OUTPUT_FOLDER'], name)
    return send_from_directory(full_path, "output.txt", as_attachment=True)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            dt = datetime.today()  
            seconds = dt.timestamp()

            os.mkdir(f"outputs/{seconds}")
            convert(file_path, f"outputs/{seconds}", request.form.to_dict())
            
            return redirect(url_for('download_file', name=seconds))
        
    return '''
    <!doctype html>
    <title>Hatching Style</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file><br/><br/>
      <label for="fname">Image Scale:</label>
      <input type="text" id="image_scale" name="image_scale" value=5><br/><br/>
      <label for="fname">Hatch Angle:</label>
      <input type="text" id="hatch_angle" name="hatch_angle" value=45><br/><br/>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
   app.run()
