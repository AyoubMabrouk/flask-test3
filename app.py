from flask import Flask, render_template, request, flash
from predictionPolySomme import predictPolySomme
import predictRejetCCE
# from prediction import courbe

app = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"
@app.route('/')
def home():
	return 'Hello, this is the home page!'


@app.route("/hello")
def index():
	flash("Choose a date")
	return render_template("index.html")

@app.route("/greet", methods=['POST', 'GET'])
def greeter():
	# courbe()
	result = predictPolySomme(float(request.form['name_input'][8:10]))
	# Format the result with two decimal places using format()
	formatted_result = format(result, ".2f")
	formatted_result_str = str(formatted_result)
	flash("For date = " + str(request.form['name_input']) + ", Sum of Montant = "+formatted_result_str)
	return render_template("index.html")



@app.route('/predictRCCE', methods=['GET', 'POST'])

def basic():
	if request.method == 'POST':
		Etat = request.form['Etat']
		IdTypeLot = request.form['IdTypeLot']
		BanquePayeur = request.form['BanquePayeur']
		MotifRejet = request.form['MotifRejet']
		y_pred = [[Etat, IdTypeLot, BanquePayeur, MotifRejet]]
		trained_model = predictRejetCCE.training_model()
		prediction_value = trained_model.predict(y_pred)
		setosa = 'the cheque was Not blocked :) '
		versicolor = 'the cheque was blocked :('
		virginica = 'other situation'
		if prediction_value == 0:
			return render_template('indexRCCE.html', setosa=setosa)
		elif prediction_value == -1:
			return render_template('indexRCCE.html', versicolor=versicolor)
		else:
			return render_template('indexRCCE.html', virginica=virginica)
	return render_template('indexRCCE.html')
