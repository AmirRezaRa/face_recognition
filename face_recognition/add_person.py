import os 
import cv2
import numpy as np
from insightface.app import FaceAnalysis


app = FaceAnalysis(name= 'buffalo_1')
app.prepare(ctx_id=-1, det_size=(640,640))

database = {}


for person_name in os.listdir('images'):
    person_path = os.path.join('images', person_name)
    
    embeddings = []
    
    for image in os.listdir(person_path):
        image_path = os.path.join(person_path, image)
        img = cv2.imread(image_path)
        if img is None:
            print('img is none')
            continue
        
        faces = app.get(img)
        
        if len(faces) == 0:
            print('No face found ...')
            continue
        
        embedding = faces[0].embedding
        embeddings.append(embedding)

    if len(embeddings) > 0 :
        mean_embedding = np.mean(embeddings, axis=0)
        database[person_name] = mean_embedding
        print(f"Added {person_name}")

np.save('output_file.npy', database)


