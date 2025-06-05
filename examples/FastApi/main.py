
from tensorflow.keras.models import load_model
model = load_model('number.keras')


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import os
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # 허용할 Origin 목록
    allow_credentials=True,
    allow_methods=["*"],              # 허용할 HTTP 메소드 (GET, POST 등)
    allow_headers=["*"],              # 허용할 HTTP 헤더
)

@app.post("/predictimg")
async def predict(request: Request):
    try:
        data = await request.json()
        img_base64 = data['img_base64']
        
        # 디코딩 및 전처리
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data)).convert('L').resize((28, 28))
        img_array = np.array(img) / 255.0
        x_test = img_array.reshape(1, 28, 28, 1)
        
        prediction = np.argmax(model.predict(x_test), axis=1)
        return {
            "prediction": int(prediction[0]),
            "error": None
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "prediction": None,
            "error": str(e),
            "duration": 1000
        })
        
@app.get("/predict")
async def predict_num(num: int):
    file_path = os.path.join('samples', str(num), f'num-{num}.png')
    if not os.path.exists(file_path):
        return {
            'prediction': None,
            'error': 'no such file name'
        }
        
    img = Image.open(file_path).convert('L')
    img_array = np.array(img) / 255
    x_test = img_array.reshape(1, 28, 28, 1)
    prediction = np.argmax( model.predict(x_test), axis=1 )
    print(prediction[0])
    print(prediction.shape)
    return {
        'prediction': int(prediction[0]),
        'error': None
    }

@app.get("/")
async def root(x: int):
    
    y = predict(x)
    
    return {"prediction": y}

@app.get('/predict/name')
async def predict_name(age: int, height: int, qqq: str):
    
    name = 'Unknown'
    
    if age < 10:
        name = 'Bob'
    else:
        name = 'Kim'
        
    if height > 200:
        name = 'Linux'
        
    return {
        'prediction': name,
        'age': age,
        'info': {
            'friends': ['song', 'park']
        }
    }
    




