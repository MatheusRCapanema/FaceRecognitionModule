import face_recognition
import cv2
import os
import pickle
from datetime import datetime, timedelta

def load_known_faces(user_data_dir):
    known_faces = []
    known_names = []

    for user_id in os.listdir(user_data_dir):
        user_path = os.path.join(user_data_dir, user_id)
        with open(os.path.join(user_path, f"{user_id}_face_encoding.pickle"), 'rb') as f:
            known_faces.append(pickle.load(f))
            known_names.append(user_id)

    return known_faces, known_names

def recognize_users(known_faces, known_names):
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    session_active = False
    last_recognition_time = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erro ao capturar vídeo.")
            break

        # Reduzir o tamanho do frame para acelerar o processo de reconhecimento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convertendo a imagem de BGR para RGB (que o face_recognition usa)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Processa apenas cada outro frame para economizar tempo
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Desconhecido"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                    last_recognition_time = datetime.now()

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Exibe os resultados
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # Verifica se a sessão ainda está ativa
        if last_recognition_time and datetime.now() - last_recognition_time > timedelta(minutes=10):
            print("Sessão expirada. Bloqueando Magic Mirror.")
            # Ação para bloquear o Magic Mirror aqui
            last_recognition_time = None

        # Aperte 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_data_dir = 'user_data'
    known_faces, known_names = load_known_faces(user_data_dir)
    recognize_users(known_faces, known_names)
