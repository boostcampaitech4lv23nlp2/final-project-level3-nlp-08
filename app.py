from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, Depends
from typing import List
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from starlette.staticfiles import StaticFiles
import datetime

import uvicorn

from src.elasticsearch.elastic import ElasticObject

    
app = FastAPI()

app.mount("/assets", app=StaticFiles(directory="assets"), name='assets')

templates = Jinja2Templates(directory='./templates')


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse('home.html', {"request": request})

@app.get("/chat", response_class=HTMLResponse)
def read_chat(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@app.get("/api/current_user")
def get_user(request: Request):
    return request.cookies.get("X-Authorization")

class RegisterValidator(BaseModel):
    username: str
    
    class Config:
        orm_mode = True
        
@app.post("/api/register")
def register_user(user: RegisterValidator, response: Response):
    response.set_cookie(key="X-Authorization", value=user.username, httponly=True)
    
    
    
class SummaryModel:
    def __init__(self) -> None:
        
        # TODO
        self.summary_output = {"summary": "제주도에 맛집을 찾고 있다. 어디서 묵는게 좋은지 생각하고 있다."}
    
    async def inference(self):
        return self.summary_output


models = SummaryModel()
elastic_connector = ElasticObject("localhost:9200")

async def summary_retrieve():
    
    summary_output = await models.inference()
    _, outputs = await elastic_connector.search(index_name="blogs", question=summary_output, topk=5)
    
    return outputs

    
class SocketManager:
    def __init__(self):
        self.active_connections: List[(WebSocket, str)] = []
        self.update_time = datetime.datetime.now()

    async def connect(self, websocket: WebSocket, user: str):
        await websocket.accept()
        self.active_connections.append((websocket, user))

    def disconnect(self, websocket: WebSocket, user: str):
        self.active_connections.remove((websocket, user))

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            await connection[0].send_json(data)
            
    def check_recommend(self):
        now_time = datetime.datetime.now()
        if ((now_time - self.update_time).seconds / 60) > 5:
            self.update_time = now_time
            return True
        return False
        
        

manager = SocketManager()

@app.websocket("/api/chat")
async def chat(websocket: WebSocket):
    sender = websocket.cookies.get("X-Authorization")
    print(sender)
    if sender:
        await manager.connect(websocket, sender)
        response = {
            "location": "chat",
            "sender": sender,
            "message": "접속하셨습니다."
        }
        messages = ""
        await manager.broadcast(response)
        try:
            while True:
                data = await websocket.receive_json()
                print(data)
                messages += data['message']
                await manager.broadcast(data)
                
                if len(messages) >= 100 or (manager.check_recommend() and len(messages) > 70):
                    outputs = await summary_retrieve()
                    
                    messages = ""
                    await manager.broadcast(outputs)
                        
                
        except WebSocketDisconnect:
            manager.disconnect(websocket, sender)
            response['message'] = "left"
            await manager.broadcast(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30001)