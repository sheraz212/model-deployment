from flask import Flask, app, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pretrainModels import get_prediction, load_img
from customtrainModel import get_predCustom, custom_open
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///imagedetails.db' 


#initializing database
db = SQLAlchemy(app)
#create Model
class Imagedetails(db.Model):
    id = db.Column(db.Integer, primary_key=True )
    picName = db.Column (db.String(100), nullable = False)
    ipAddress = db.Column (db.String(100), nullable = False)
    date = db.Column (db.DateTime, default = datetime.utcnow)
    
    def __repr__(self):
        return  '<IP Address %r>' % self.ipAddress

ALLOWED_IMGTYPES = {'png','jpg','jpeg'}
def allowed_FILETYPE(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_IMGTYPES
 
ALL_MODELS = ['ECCV16' , 'SIGGRAPH17' , 'PIXUMMODEL']

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if file is None or file.filename == "":
            return jsonify({'ERROR':'There is NO FILE'})
        if not allowed_FILETYPE(file.filename):
            return jsonify({'ERROR':'INVALID FILE TYPE ONLY ALLOWED ARE '})
        
        
        
        try:
            image_in = load_img(file.stream)
            image_customIn = custom_open(file.stream)
            
            custom_pred = get_predCustom(image_customIn)
            prediction1 , prediction2 = get_prediction (image_in)
            
            ip_address = request.remote_addr
            m1 = './images/' + ip_address + '0' + file.filename
            m2 = './images/'+ ip_address +'1' + file.filename
            m3 = './images/'+ ip_address +'2' + file.filename
            
            image_loc1 = 'http://localhost:5000/images/' + ip_address + '0' + file.filename
            image_loc2 = 'http://localhost:5000/images/' + ip_address + '1' + file.filename
            image_loc3 = 'http://localhost:5000/images/' + ip_address + '2' + file.filename
            
            plt.imsave(m1, prediction2)
            plt.imsave(m2, prediction1)
            plt.imsave(m3, custom_pred)
            
            retDICT = {}
            #for i in range ALL_MODELS.len:
            retDICT[ALL_MODELS[0]] = image_loc1
            retDICT[ALL_MODELS[1]] = image_loc2
            retDICT[ALL_MODELS[2]] = image_loc3
            
                    
            newEntry = Imagedetails(picName=(image_loc1),ipAddress = ip_address )
            newEntry1 = Imagedetails(picName=(image_loc2),ipAddress = ip_address )
            newEntry2 = Imagedetails(picName=(image_loc3),ipAddress = ip_address )
            db.session.add(newEntry)
            db.session.commit()
            db.session.add(newEntry1)
            db.session.commit()
            db.session.add(newEntry2)
            db.session.commit()
            
            return jsonify(retDICT)
        
        except:
             return jsonify({'ERROR':'Error in predictions'})
   
    

if __name__ == '__main__':
    app.debug = True
    app.run()


#Changing ENVIRONMENT
#.\pytorAPI\Scripts\activte.ps1
