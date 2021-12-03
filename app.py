from flask import *  
from tensorflow import keras
import numpy as np
from keras.preprocessing import image
import os
app = Flask(__name__)  

@app.route('/')  
def upload():  
    return render_template("index.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        cnn = keras.models.load_model(os.getcwd()+"/model/road_damage.h5")
        test_image = image.load_img(f.filename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        if result[0][0] == 1:
         prediction = 'Damaged Road'
        else:
         prediction = 'Clean Road'  
        os.remove(os.getcwd() +"/"+ f.filename)
        return render_template("index.html", message = 'Prediction: '+prediction)  
  
if __name__ == '__main__':  
    app.run(debug = True)  