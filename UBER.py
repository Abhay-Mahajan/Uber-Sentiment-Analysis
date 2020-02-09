import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from flask import Flask,request,render_template,url_for

df = pd.read_csv(r'UBER.csv', encoding = "ISO-8859-1")

df['sentiment'] = np.where(df['ride_rating'] > 3 , 'Positive' , 'Negative')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

	X = df['ride_review']
	y = df['sentiment']
	tf = TfidfVectorizer()
	X = tf.fit_transform(X)

	X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.30, random_state=42) 

	# Create a Logistic Regression model
	logistic_regression = LogisticRegression(solver='lbfgs', max_iter=1000)
	logistic_regression.fit(X, y)
	logistic_regression.score(X_test, y_test)
	

	if request.method == 'POST':
		ride_review = request.form['text']
		data = [ride_review]
		vect = tf.transform(data).toarray()
		pred = logistic_regression.predict(vect)

		return render_template("index.html", sentiment =pred)

if __name__ == '__main__':
	app.run(debug=True)