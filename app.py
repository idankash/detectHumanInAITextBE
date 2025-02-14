#https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/
#https://www.youtube.com/watch?v=qbLc5a9jdXo&ab_channel=CalebCurry
#https://stackoverflow.com/questions/26368306/export-is-not-recognized-as-an-internal-or-external-command
#python3 -m venv .venv
#source .venv/bin/activate
#
#pip freeze > requirements.txt
#$env:FLASK_APP="application.py" #set FLASK_APP=application.py # export FLASK_APP=application.py 
#set FLASK_ENV=development #export FLASK_ENV=production
#flask run #flask run --host=0.0.0.0

#pip install torchvision

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas
from human_text_detect import detect_human_text

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Hello'

@app.route('/detectHumanInAIText/checkText', methods=['POST'])
def check_text():

    # Get data
    print('Get data')
    data = request.get_json()
    text = data.get('text')
    model_name = data.get('model')
    topic = data.get('topic')
    
    # Validate data
    print('Validate data')
    answer = validate_data(text, model_name, topic)
    if answer != '':
        return jsonify({'answer': answer}), 400
    
    topic = check_topic(topic)
    answer = detect_human_text(model_name, topic, text)

    return jsonify({'answer': answer})

def validate_data(text, model_name, topic):
    if text is None or text == '':
        return 'Text is missing'

    if model_name is None or model_name == '':
        return 'Model name is missing'

    if topic is None or topic == '':
        return 'Topic is missing'
    
    if model_name not in ['GPT2XL', 'PHI2']:
        return f'Model {model_name} not supported'
    
    if topic not in ['Characters', 'Locations', 'Nature', 'Video games', 'Series', 'Movies', 'War']:
        return f'Topic {topic} not supported'
    
    return ''

def check_topic(topic):
    topic_dict = {
        'Characters': 'characters',
        'Locations': 'locations',
        'Nature': 'nature',
        'Video games': 'video_games_series_movies',
        'Series': 'video_games_series_movies',
        'Movies': 'video_games_series_movies',
        'War': 'war'
    }

    return topic_dict[topic]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT variable or default to 5000
    app.run(host="0.0.0.0", port=port)
