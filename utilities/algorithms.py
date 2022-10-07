import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

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
				 				   				  	 'desc' : 'Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses.'}   
				 				   },
				 	'link' : "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
			},
		 	'tree' : {'name' : 'Decision Tree Classifier',
		 			 'instance' : DecisionTreeClassifier(),
				   	  'parameters' : {'max_depth' : {'value' :[None, 2, 3, 4, 5, 10],
					 				   				 'type':int,
					 				   				 'desc' : 'The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.'},
				 					  'max_leaf_nodes': {'value' :[None, 2, 3, 4, 5, 10],
				 				   				  		 'type':int,
				 				   				  		 'desc' : 'Grow a tree with max_leaf_nodes in best-first fashion. \n Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.'}
				 					  },
				 	  'link' : "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"
				   },
		 	'forest' : {'name' : 'Random Forest',
		 			 'instance' : RandomForestClassifier(),
				   	 'parameters' : {'n_estimators' : {'value' :[10, 20, 30, 50, 100, 200],
				 				   				  	   'type':int,
				 				   				  	   'desc' : 'The number of Decision trees in the forest.'},
				   	 				 'max_depth' : {'value' :[2, 3, 4, 5, 10],
					 				   				  'type':int,
					 				   				  'desc' : 'The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.'},
				 					 'max_leaf_nodes': {'value' :[2, 3, 4, 5, 10],
				 				   				  		'type':int,
				 				   				  		'desc' : 'Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.'}
				 					  },
				 	 'link' : "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
				   },
			'svm' : {'name' : 'Support Vector Machine',
					 'instance' : SVC(),
				   	 'parameters' : {'C' : {'value' :[0.01, 0.1, 1, 10, 100],
											'type':float,
											'desc' : 'Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.'},
						   	 		'degree' : {'value': [1, 3, 5 ,10],
						   	 					'type':int,
						   	 					'desc' : 'Degree of the polynomial kernel function (‘poly’).'},
				   	 				'kernel' : {'value': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
				   	 				 			'type':str,
				   	 				 			'desc' : 'The function of kernel is to take data as input and transform it into the required form.\n The kernel function is what is applied on each data instance to map the original non-linear observations into a higher-dimensional space in which they become separable'}
				   	 				 },
				   	 	'link' : "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
				   	 },
			'boost' : {'name' : 'Gradient Boosting',
					 'instance' : GradientBoostingClassifier(),
				   	 'parameters' : {'loss' :{'value':['deviance','exponential'],
			   	 							  'type':str,
			   	 							  'desc' : ''},
				   	 				'learning_rate' : {'value':[0.001, 0.01, 0.1, 1, 5],
				   	 									'type':float,
				   	 									'desc' : 'the gradient of the training loss is used to change the target variables for each successive tree \n loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.'},
				   	 				'n_estimators' : {'value':[10, 20, 50, 100 ,200],
				   	 								  'type':int,
				   	 								  'desc' : 'The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.'},
				   	 				'criterion' : {'value':['friedman_mse' ,'mse', 'mae'],
				   	 								'type':str,
				   	 								'desc' : 'The function to measure the quality of a split. Supported criteria are ‘friedman_mse’ for the mean squared error with improvement score by Friedman, ‘mse’ for mean squared error, and ‘mae’ for the mean absolute error. '}
				   	 				},
				   	 'link' : "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"
				   	 },
			'nn' : {'name' : 'Neural Network',
					 'instance' : MLPClassifier(),
				   	 'parameters' : {'activation' :{'value':['identity', 'logistic', 'tanh', 'relu'],
				   	 							'type':str,
				   	 							'desc' : 'Activation function for the hidden layer.\n 1)‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x \n 2)‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).\n 3)‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x). \n 4)‘relu’, the rectified linear unit function, returns f(x) = max(0, x)'},
				   	 				'optimizer' : {'value':['sgd', 'adam'],
				   	 							'type':str,
				   	 							'desc' : 'Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses.'},
				   	 				'learning_rate_init' : {'value':[0.0001, 0.001, 0.01, 0.10, 1.00, 10.00],
				   	 							'type':float,
				   	 							'desc' : 'The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.'}
				   	 				},
				   	 'link' : "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"
				   	 },
			'nb' : {'name' : 'Naive Bayes',
					 'instance' : GaussianNB(),
				   	 'parameters' : {'var_smoothing' : {'value' : [1e-9],
				   	 									'type':float,
				   	 									'desc' : 'Portion of the largest variance of all features that is added to variances for calculation stability.'}
				   	 				},
				   	 'link' : "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html"
				   	 }
		}