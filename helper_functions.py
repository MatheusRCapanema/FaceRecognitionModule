import os
import pickle

def load_user_encodings(user_data_dir):
    user_encodings = {}
    for user_id in os.listdir(user_data_dir):
        user_path = os.path.join(user_data_dir, user_id)
        encoding_file = os.path.join(user_path, f"{user_id}_face_encoding.pickle")
        if os.path.isfile(encoding_file):
            with open(encoding_file, 'rb') as f:
                user_encodings[user_id] = pickle.load(f)
    return user_encodings

def save_user_encoding(user_id, user_encoding, user_data_dir):
    user_path = os.path.join(user_data_dir, user_id)
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    encoding_file = os.path.join(user_path, f"{user_id}_face_encoding.pickle")
    with open(encoding_file, 'wb') as f:
        pickle.dump(user_encoding, f)

# Outras funções auxiliares conforme necessário
