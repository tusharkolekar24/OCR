from tensorflow.keras.models import model_from_json
import json

# load json and create model
json_file = open(r'artifacts\model.json', 'r')
loaded_model_json = json_file.read()
# json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(r"artifacts\model.h5")
print("Loaded model from disk")

encoder_json_file = open(r'artifacts\label_encoder.json', 'r')
encoder_json = json.loads(encoder_json_file.read())