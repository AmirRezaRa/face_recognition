import cv2
import numpy as np
from insightface.app import FaceAnalysis

def cosine_similarity(a,b):
    return np.dot(a,b)/ (np.linalg.norm(a)* np.linalg.norm(b))


database = np.load('face_detection/output_file.npy', allow_pickle=True).item()

app = FaceAnalysis(name='buffalo_1')
app.prepare(ctx_id=-1, det_size =(640,640))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = app.get(frame)
    
    for face in faces:
        emb = face.embedding
        
        name = 'unknown'
        best_score = 0
        THERESHOLD = 0.5
        
        for person_name, db_emb in database.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                name = person_name
        
        if best_score < THERESHOLD :
            name = 'unknown'
        
        x1,y1, x2,y2 = map(int, face.bbox)
        if name == 'unknown':
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            
        
        cv2.imshow('face_recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
                
