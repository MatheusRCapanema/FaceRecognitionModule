import os
import cv2
import face_recognition
import pickle

def register_new_user(db_dir, user_id, frame):
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    user_path = os.path.join(db_dir, user_id)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    embeddings = face_recognition.face_encodings(frame)
    if embeddings:
        filepath = os.path.join(user_path, f"{user_id}_face_encoding.pickle")
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings[0], f)
        print(f"Usuário {user_id} registrado com sucesso.")
    else:
        print("Nenhum rosto detectado na imagem.")

def capture_new_user(db_dir):
    user_id = input("Digite o ID do usuário para o registro: ")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar vídeo.")
            continue
        cv2.imshow("Pressione 'c' para capturar uma imagem para o registro.", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            register_new_user(db_dir, user_id, frame)
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    db_dir = 'user_data'
    capture_new_user(db_dir)

if __name__ == "__main__":
    main()
