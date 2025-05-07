from tensorflow.keras.models import model_from_json
import json


def get_model_info(model_name):
    
    

    if model_name=='CNN':
        json_file_path = r'artifacts\model.json'
        weights_path   = r"artifacts\model.h5"
        encoder_path   = r'artifacts\label_encoder_{}.json'.format(model_name)

    if model_name!='CNN': 
        json_file_path = r'artifacts\{}.json'.format(model_name)
        weights_path   = r"artifacts\{}.h5".format(model_name)
        encoder_path   = r'artifacts\label_encoder_{}.json'.format(model_name)
    
    # load json and create model
    json_file = open(json_file_path, 'r')
    loaded_model_json = json_file.read()
    # json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weights_path)
    print(f"{model_name} Loaded from disk")

    encoder_json_file = open(encoder_path, 'r')
    encoder_json = json.loads(encoder_json_file.read())

    return loaded_model, encoder_json