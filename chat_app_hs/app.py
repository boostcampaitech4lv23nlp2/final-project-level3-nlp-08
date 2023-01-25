from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, Depends
from typing import List
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from starlette.staticfiles import StaticFiles
import datetime

import uvicorn

    
app = FastAPI()

app.mount("/assets", app=StaticFiles(directory="assets"), name='assets')

templates = Jinja2Templates(directory='./')


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
    
    
class ElasticTest:
    def __init__(self) -> None:
        self.elastic_outputs = {"location":"recommend", "message": "이것도 읽어봐라 멍멍!!", "source": [{"url":"naver.com", "title": "돼지고기 맛집"}, {"url": "naver.com", "title": "소고기 맛집"}]}
    
    async def search(self):
        return self.elastic_outputs
    
    
class SummaryModel:
    def __init__(self) -> None:
        
        # TODO time-time
        self.summary_output = {"location": "summary", "message": "지금까지 한 대화를 요약해봤어"+"<br>"+"제주도에 가서 흑돼지를 먹고 이쁜 카페에 가자."}
    
    async def inference(self):
        return self.summary_output
    
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
elastic = ElasticTest()
models = SummaryModel()

@app.websocket("/api/chat")
async def chat(websocket: WebSocket):
    sender = websocket.cookies.get("X-Authorization")
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
                messages += (data['message'] + " ")
                await manager.broadcast(data)
                
                if len(messages) >= 200 or (manager.check_recommend() and len(messages) > 150):
                    elastic_outputs = await elastic.search()
                    summary_output = await models.inference()
                    
                    messages = ""
                    await manager.broadcast(elastic_outputs)
                    await manager.broadcast(summary_output)
                    
                
                        
                
        except WebSocketDisconnect:
            manager.disconnect(websocket, sender)
            response['message'] = "left"
            await manager.broadcast(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")