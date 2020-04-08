from flask import *
import os

import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

def clean_and_predict(matches, ranking, final, rf):

    # Initialization of auxiliary list for data cleaning
    positions = []

    # Loop to retrieve each team's position according to ICC ranking
    for match in matches:
        positions.append(ranking.loc[ranking['Team'] == match[0],'Position'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == match[1],'Position'].iloc[0])
    
    # Creating the DataFrame for prediction
    pred_set = []

    # Initializing iterators for while loop
    i = 0
    j = 0

    # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
    while i < len(positions):
        dict1 = {}

        # If position of first team is better then this team will be the 'Team_1' team, and vice-versa
        if positions[i] < positions[i + 1]:
            dict1.update({'Team_1': matches[j][0], 'Team_2': matches[j][1]})
        else:
            dict1.update({'Team_1': matches[j][1], 'Team_2': matches[j][0]})

        # Append updated dictionary to the list, that will later be converted into a DataFrame
        pred_set.append(dict1)
        i += 2
        j += 1
        
        # Convert list into DataFrame
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set
    
    
    
    # Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

    # Add missing columns compared to the model's training dataset
    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]
    
    pred_set = pred_set.drop(['Winner'], axis=1)

    # Predict!
    predictions = rf.predict(pred_set)
    return predictions

# Use pickle to load in the pre-trained model.
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('main.html')

@app.route('/predict', methods=['POST'])

def predict():
    # if request.method == 'GET':
    #    return(render_template('main.html'))
    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        ranking = pd.read_csv('icc_rankings.csv') 
        final =  pd.read_csv('final.csv') 
        finals = [(team1, team2)]
      
        prediction = clean_and_predict(finals, ranking, final, model)[0]
       # prediction = model.predict(input_variables)[0]
        return render_template('main.html',
                                     original_input={'team1':team1,
                                                     'team2':team2,
                                                     },
                                     result="winner is {}".format(prediction)
                                     )

if __name__ == "__main__":
	app.run(debug=True)