from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import matplotlib.pyplot as plt
from utilities import algorithms
from utilities.algorithms import *
from os import urandom
import json
from joblib import load, dump
import random
from flask_bcrypt import check_password_hash
from flask_login import current_user

## For user authentication

from flask_login import (
    UserMixin,
    login_user,
    LoginManager,
    current_user,
    logout_user,
    login_required,
)
from extensions import db, migrate, bcrypt, login_manager

app = Flask(__name__)

app.secret_key = urandom(24)


app = Flask(__name__)
app.secret_key = urandom(24)

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db.init_app(app)
migrate.init_app(app, db)
bcrypt.init_app(app)
login_manager.init_app(app)
login_manager.session_protection = "strong"
login_manager.login_view = "login"
login_manager.login_message_category = "info"


# migrate.init_app(app, db)
# bcrypt.init_app(app)


# @login_manager.user_loader
# def load_user(user_id):
# 	new_user = User(username='jdoe', email='jdoe@example.com', pwd='password123')
# 	db.db.session.add(new_user)
# 	return User.query.get(int(user_id))
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# app.config['SECRET_KEY'] = urandom(24)
from forms import *
df = pd.read_csv('Dataset\\diabetes_data_upload.csv')
content_api = 'api\\indexcontent.json'
feature_api = 'api\\features.json'
file1 = open(content_api, )
file2 = open(feature_api, )
FEATURES = df.columns
SPLIT_DATA = False
ENCODED = False
MODEL = None
STEP = 1
CONTENT = json.load(file1)
FEATURE_API = json.load(file2)
ALGORITHMS = algorithms.ALGORITHMS
features_train,features_test,labels_train,labels_test = None,None,None,None
train_score, test_score, test_pred = None, None, None


@app.route('/')
@login_required
def index():
	return render_template('index.html', algorithms=ALGORITHMS, content=CONTENT, step=STEP) 

@app.route('/visualize')
@login_required
def visualize():
	global ENCODED
	global FEATURES
	data = df
	return render_template('visualize.html', Dataset=data, encoded=ENCODED, features=FEATURES)

@app.route('/visualize/encode')
@login_required
def encode():
	global ENCODED
	global STEP
	from sklearn.preprocessing import LabelEncoder
	columns = df.columns[1:]
	for label in columns:
		df[label] = LabelEncoder().fit_transform(df[label])
	ENCODED = True
	STEP = 2
	flash('Data Encoded!!!')
	return redirect('/visualize')

@app.route('/split-data', methods=['POST', 'GET'])
@login_required
def split():
	global SPLIT_DATA
	global STEP
	if STEP < 2:
		return render_template('404.html', code=500, e=f"404 Not Found : You Cannot Split Data Right Now! Complete earlier steps! You are On step {STEP} / 6")
	global features_train,features_test,labels_train,labels_test
	features, labels = df.drop(['class'], axis=1), df['class']
	test_size = 0.2
	random_state = 0
	shuffle = True
	if request.method == 'POST':
		split = request.form
		test_size = float(split['test_size'])
		random_state = int(split['random_state'])
		shuffle = bool(split['shuffle'])
		features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = test_size, random_state = random_state, shuffle = shuffle)
		SPLIT_DATA = True
		STEP = 3
		flash('Data Splitted Successfully')

	return render_template('split.html', split=SPLIT_DATA, features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test)
@login_required	
@app.route('/reset-split')
def reset():
	global ALGORITHMS
	global STEP
	global SPLIT_DATA
	global features_train,features_test,labels_train,labels_test
	SPLIT_DATA = False
	if STEP < 2:
		return render_template('404.html', code=500, e=f"404 Not Found : Complete earlier steps!\nYou are On step {STEP} / 6")
	STEP = 2
	features_train,features_test,labels_train,labels_test = None,None,None,None
	flash('Train Test Split Resetted')

	return redirect('/split-data')

@app.route('/<algorithm>/parameters')
@login_required
def parameters(algorithm):
	if STEP < 3:
		return render_template('404.html', code=500, e=f"404 Not Found : You Cannot Decide Algorithm Right Now!! Complete earlier steps!\nYou are On step {STEP} / 6")
	return render_template('parameters.html', name=algorithm, algorithms=ALGORITHMS)

@app.route('/<model>/build-model', methods = ['GET', 'POST'])
@login_required
def build_model(model):
	global MODEL
	global STEP
	if STEP < 3:
		return render_template('404.html', code=500, e=f"404 Not Found : You cannot build Model Right Now!! Complete earlier steps!\nYou are On step {STEP} / 6")
	session['model'] = model
	if request.method == 'POST':
		params = request.form
		for key in ALGORITHMS[model]['parameters'].keys():
			value = params[key]
			if ALGORITHMS[model]['parameters'][key]['type'] == int:
				ALGORITHMS[model]['parameters'][key] = int(value)
			elif ALGORITHMS[model]['parameters'][key]['type'] == str:
				ALGORITHMS[model]['parameters'][key] = value
			elif ALGORITHMS[model]['parameters'][key]['type'] == float:
				ALGORITHMS[model]['parameters'][key] = float(value)
			elif ALGORITHMS[model]['parameters'][key]['type'] == bool:
				ALGORITHMS[model]['parameters'][key] = bool(value)
		MODEL = ALGORITHMS[model]['instance']
		if MODEL == ALGORITHMS['knn']['instance']:
			MODEL.set_params(n_neighbors=ALGORITHMS[model]['parameters']['n_neighbors'], 
				algorithm=ALGORITHMS[model]['parameters']['algorithm'],
				p=ALGORITHMS[model]['parameters']['p'])

		elif MODEL == ALGORITHMS['tree']['instance']:
			MODEL.set_params(max_depth=ALGORITHMS[model]['parameters']['max_depth'], 
				max_leaf_nodes=ALGORITHMS[model]['parameters']['max_leaf_nodes'])

		elif MODEL == ALGORITHMS['forest']['instance']:
			MODEL.set_params(n_estimators=ALGORITHMS[model]['parameters']['n_estimators'],
				max_depth=ALGORITHMS[model]['parameters']['max_depth'], 
				max_leaf_nodes=ALGORITHMS[model]['parameters']['max_leaf_nodes'])

		elif MODEL == ALGORITHMS['boost']['instance']:
			MODEL.set_params(loss=ALGORITHMS[model]['parameters']['loss'], 
				learning_rate=ALGORITHMS[model]['parameters']['learning_rate'],
				n_estimators=ALGORITHMS[model]['parameters']['n_estimators'],
				criterion=ALGORITHMS[model]['parameters']['criterion'])

		elif MODEL == ALGORITHMS['logreg']['instance']:
			MODEL.set_params(C=ALGORITHMS[model]['parameters']['C'], 
				max_iter=ALGORITHMS[model]['parameters']['max_iter'],
				solver=ALGORITHMS[model]['parameters']['solver'])

		elif MODEL == ALGORITHMS['svm']['instance']:
			MODEL.set_params(C=ALGORITHMS[model]['parameters']['C'], 
				degree=ALGORITHMS[model]['parameters']['degree'],
				kernel=ALGORITHMS[model]['parameters']['kernel'],
				probability=True)

		elif MODEL == ALGORITHMS['nn']['instance']:
			MODEL.set_params(activation=ALGORITHMS[model]['parameters']['activation'], 
				solver=ALGORITHMS[model]['parameters']['optimizer'],
				learning_rate_init=ALGORITHMS[model]['parameters']['learning_rate_init'],
				max_iter=500)

		elif MODEL == ALGORITHMS['nb']['instance']:
			MODEL.set_params(var_smoothing=ALGORITHMS[model]['parameters']['var_smoothing'])
		STEP = 4
	return render_template('build_model.html', algo=ALGORITHMS[model], model=model, createdModel=MODEL)

@app.route('/reset-parameters')
@login_required
def reset_params():
	global ALGORITHMS
	if STEP < 4:
		return render_template('404.html', code=500, e=f"404 Not Found : Complete earlier steps!\nYou are On step {STEP} / 6")
	ALGORITHMS = {
				'knn' : {'name' : 'K-Nearest Neighbors',
						 'instance' : KNeighborsClassifier(),
					   	 'parameters' : {'n_neighbors' : {'value' : [2, 3, 4, 5, 6, 7],
					   	 								  'type' : int,
					   	 								  'desc' : '‘n_neighbors‘ are the number of neighbors that will vote for the class of the target point.\nDefault number is 5. An odd number is preferred to avoid any tie.'},
					 				   'algorithm' : {'value' :['auto', 'ball_tree', 'brute', 'kd_tree'],
					 				   				  'type':str,
					 				   				  'desc' : 'Parameter ‘algorithm‘ is for\nselecting the indexing data structure that will be used for speeding up neighborhood search.\n Value of ‘auto‘ leaves it to algorithm to make the best choice among the three'},
									   'p' : {'value':[1, 2],
									   		  'type':int,
									   		  'desc' : 'When parameter ‘p‘ is 2, it is the same as euclidean distance and when parameter ‘p‘ is 1, it is Manhattan distance.'}
					 				   },
					 	'link' : "https://ashokharnal.wordpress.com/tag/kneighborsclassifier-explained/#:~:text='n_neighbors'%20are%20the%20number%20of,point%3B%20default%20number%20is%205.&text=If%20the%20weight%20is%20'distance,those%20who%20are%20farther%20away."
					  },
				'logreg' : {'name' : 'Logistic Regression',
						 'instance' : LogisticRegression(),
					   	 'parameters' :{'C' : {'value' :[0.01, 0.1, 1, 10, 100],
					 				   			'type':float,
					 				   			'desc' : "The trade-off parameter of logistic regression that determines the strength of the regularization is called C, and higher values of C correspond to less regularization"},
					 				   'max_iter' : {'value' : [10, 20, 50, 100, 200],
					   	 							 'type' : int,
					   	 							 'desc' : 'Maximum number of iterations taken for the solver to converge'},
					 				   'solver' : {'value' :['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
					 				   				  	 'type':str,
					 				   				  	 'desc' : ''}
					 				   },
					 	'link' : ""
				},
			 	'tree' : {'name' : 'Decision Tree Classifier',
			 			 'instance' : DecisionTreeClassifier(),
					   	  'parameters' : {'max_depth' : {'value' :[2, 3, 4, 5, 10],
						 				   				 'type':int,
						 				   				 'desc' : ''},
					 					  'max_leaf_nodes': {'value' :[2, 3, 4, 5, 10],
					 				   				  		 'type':int,
					 				   				  		 'desc' : ''}
					 					  },
					 	  'link' : ""
					   },
			 	'forest' : {'name' : 'Random Forest',
			 			 'instance' : RandomForestClassifier(),
					   	 'parameters' : {'n_estimators' : {'value' :[10, 20, 30, 50, 100, 200],
					 				   				  	   'type':int,
					 				   				  	   'desc' : ''},
					   	 				 'max_depth' : {'value' :[2, 3, 4, 5, 10],
						 				   				  'type':int,
						 				   				  'desc' : ''},
					 					 'max_leaf_nodes': {'value' :[2, 3, 4, 5, 10],
					 				   				  		'type':int,
					 				   				  		'desc' : ''}
					 					  },
					 	 'link' : ""
					   },
				'svm' : {'name' : 'Support Vector Machine',
						 'instance' : SVC(),
					   	 'parameters' : {'C' : {'value' :[0.01, 0.1, 1, 10, 100],
												'type':float,
												'desc' : ''},
							   	 		'degree' : {'value': [1, 3, 5 ,10],
							   	 					'type':int,
							   	 					'desc' : ''},
					   	 				'kernel' : {'value': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
					   	 				 			'type':str,
					   	 				 			'desc' : ''}
					   	 				 },
					   	 	'link' : ""
					   	 },
				'boost' : {'name' : 'Gradient Boosting',
						 'instance' : GradientBoostingClassifier(),
					   	 'parameters' : {'loss' :{'value':['deviance','exponential'],
				   	 							  'type':str,
				   	 							  'desc' : ''},
					   	 				'learning_rate' : {'value':[0.001, 0.01, 0.1, 1, 5],
					   	 									'type':float,
					   	 									'desc' : ''},
					   	 				'n_estimators' : {'value':[10, 20, 50, 100 ,200],
					   	 								  'type':int,
					   	 								  'desc' : ''},
					   	 				'criterion' : {'value':['friedman_mse' ,'mse', 'mae'],
					   	 								'type':str,
					   	 								'desc' : ''}
					   	 				},
					   	 'link' : ""
					   	 },
				'nn' : {'name' : 'Neural Network',
						 'instance' : MLPClassifier(),
					   	 'parameters' : {'activation' :{'value':['identity', 'logistic', 'tanh', 'relu'],
					   	 							'type':str,
					   	 							'desc' : ''},
					   	 				'optimizer' : {'value':['sgd', 'adam'],
					   	 							'type':str,
					   	 							'desc' : ''},
					   	 				'learning_rate_init' : {'value':[0.0001, 0.001, 0.01, 0.10, 1.00, 10.00],
					   	 							'type':float,
					   	 							'desc' : ''}
					   	 				},
					   	 'link' : ""
					   	 },
				'nb' : {'name' : 'Naive Bayes',
						 'instance' : GaussianNB(),
					   	 'parameters' : {'var_smoothing' : {'value' : [1e-9],
					   	 									'type':float,
					   	 									'desc' : ''}
					   	 				},
					   	 'link' : ""
					   	 }
			}
	return redirect('/'+session['model']+'/parameters')

@app.route('/evaluate-model')
@login_required
def eval():
	global ALGORITHMS
	global MODEL
	global STEP
	global train_score, test_score, test_pred
	if STEP < 4:
		return render_template('404.html', code=500, e=f"404 Not Found : You cannot Evaluate Model Right Now!! Complete earlier steps!\nYou are On step {STEP} / 6")
	filename = 'static\\confusion_matrix\\' + session['model'] + '.png'
	MODEL.fit(features_train, labels_train)
	train_score = MODEL.score(features_train, labels_train)
	test_score = MODEL.score(features_test, labels_test)
	test_pred = MODEL.predict(features_test)
	plot_confusion_matrix(MODEL, features_test, labels_test)
	plt.savefig(filename)
	STEP = 5
	return redirect('/evaluate')

@app.route('/evaluate')
@login_required
def evaluate():
	if STEP < 4:
		return render_template('404.html', code=500, e=f"404 Not Found : You cannot Evaluate Model Right Now!! Complete earlier steps!\nYou are On step {STEP} / 6")
	return render_template('evaluate.html', train=train_score*100, test=test_score*100, algorithm=ALGORITHMS)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def prediction():
	global MODEL
	global STEP
	if STEP < 5:
		return render_template('404.html', code=500, e=f"404 Not Found : You cannot Predict Using Model Right Now!! Complete earlier steps!\nYou are On step {STEP} / 6")
	STEP = 6
	columns = features_train.columns
	best_model = load('api\\best.pkl')
	user_values = []
	if request.method == 'POST':
		details = request.form
		for key in columns:
			user_values.append(int(details[key]))
		prediction = MODEL.predict([user_values])
		probability = MODEL.predict_proba([user_values])

		return render_template("predict.html", columns=columns, prediction=prediction, confidence=str(max(probability[0])*100), api=FEATURE_API)
	return render_template("predict.html", columns=columns, prediction=None, api=FEATURE_API)

@app.route('/download-model')
@login_required
def download_model():
	if STEP < 6:
		return render_template('404.html', code=500, e=f"404 Not Found : You cannot Download Model Right Now!! Complete earlier steps!\nYou are On step {STEP} / 6")
	dump(MODEL, 'static\\models\\'+session['model']+'.pkl')
	return render_template('download.html', algo=ALGORITHMS, name=session['model'])

@app.errorhandler(404)
def pagenotfound(e):
	return render_template('404.html', code=404, e=e)



# Login route
@app.route("/login/", methods=("GET", "POST"), strict_slashes=True)
def login():
    form = login_form()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(email=form.email.data).first()
            if check_password_hash(user.pwd, form.pwd.data):
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash("Invalid Username or password!", "danger")
        except Exception as e:
            flash(e, "danger")
    return render_template("auth.html",form=form,page_type="Login")

# Register route
@app.route("/register/", methods=("GET", "POST"), strict_slashes=True)
def register():
    form = register_form()
    
    if form.validate_on_submit():
        try:
            email = form.email.data
            pwd = form.pwd.data
            username = form.username.data
            
            newuser = User(
                username=username,
                email=email,
                pwd=bcrypt.generate_password_hash(pwd),
            )
    
            db.session.add(newuser)
            db.session.commit()
            flash(f"Account Succesfully created", "success")
            return redirect(url_for("login"))

        except Exception as e:
            flash(e, "danger")

    return render_template("auth.html",form=form,page_type="Register")
@app.route("/logout/", methods=("GET", "POST"), strict_slashes=True)
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
if __name__ == '__main__':
	app.run(debug=False, port=4000)
