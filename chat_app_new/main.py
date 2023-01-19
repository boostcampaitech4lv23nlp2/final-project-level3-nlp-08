from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, Request, Response, Depends
)
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from mongodb import get_nosql_db, connect_to_mongo, close_mongo_connection
from config import MONGODB_NAME
from bson import ObjectId
from datetime import datetime
from Elastic import elastic

import summary
import logging

app = FastAPI()
es = elastic.ElasticObject("localhost:9200")

# locate templates
templates = Jinja2Templates(directory="templates")


@app.on_event('startup')
async def startup_event():
    await connect_to_mongo()
    client = await get_nosql_db()
    db = client[MONGODB_NAME]

    try:
        message_collection = db.messages
    except pymongo.errors.CollectionInvalid as e:
        logging.info(e)
        pass

@app.on_event('shutdown')
async def shutdown_event():
    await close_mongo_connection()

@app.get("/")
def get_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/chat")
def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


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


class SocketManager:
    def __init__(self):
        self.active_connections: List[(WebSocket, str)] = []

    async def connect(self, websocket: WebSocket, user: str):
        await websocket.accept()
        self.active_connections.append((websocket, user))

    def disconnect(self, websocket: WebSocket, user: str):
        self.active_connections.remove((websocket, user))

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            await connection[0].send_json(data)

manager = SocketManager()

@app.websocket("/api/chat")
async def chat(websocket: WebSocket, client: AsyncIOMotorClient = Depends(get_nosql_db)):
    sender = websocket.cookies.get("X-Authorization")
    db = client[MONGODB_NAME]
    collection = db.messages
    
    if sender:
        await manager.connect(websocket, sender)
        response = {
            "sender": sender,
            "message": "님이 접속하셨습니다."
        }
        await manager.broadcast(response)
        print(response)
        try:
            while True:
                data = await websocket.receive_json()
                res = await stack_message(data, collection)
                messages = await get_messages()
                message_list = get_message_list(messages)
                context = ''
                await manager.broadcast(data)

                if len(message_list) == 10:
                    #대화가 15번 오가면 해당 대화를 요약해주고, DB에서 쌓인 메세지를 삭제한다.
                    context = '<s>' + messages[0].message
                    context = "</s> <s>".join(message_list)
                    context = context + '</s>'
                    summary_context = summary.summarize(context)

                    notify_data = {'sender': 'Golden Retriever', 'message': '지금까지 나눈 대화 내용을 정리해봤어!'}
                    await manager.broadcast(notify_data)

                    summary_data = {'sender': 'Golden Retriever', 'message': summary_context}
                    await manager.broadcast(summary_data)
                    
                    elastic_list = es.search('naver_docs', summary_context)
                    titles, urls = get_elastic_list(elastic_list)

                    notify_data_2 = {'sender': 'Golden Retriever', 'message': '이런 내용이 필요할 것 같은데, 어때?'}
                    await manager.broadcast(notify_data_2)
                    elastic_data = {'sender': 'Golden Retriever', 'message': titles[0], 'url': urls[0], 'hyperlink': 0}
                    await manager.broadcast(elastic_data)

                    collection.delete_many({})
                
                
        except WebSocketDisconnect:
            manager.disconnect(websocket, sender)
            response['message'] = "님이 퇴장하셨습니다."
            await manager.broadcast(response)

class Message(BaseModel):
    username: str
    message: str = None

class MessageInDB(Message):
    _id: ObjectId
    timestamp: datetime = Field(default_factory=datetime.utcnow)

async def stack_message(data, collection):
    messages = {}
    messages['username'] = data['sender']
    messages['message'] = data['message']

    dbmessage = MessageInDB(**messages)
    response = await collection.insert_one(dbmessage.dict())

async def get_messages():
    client = await get_nosql_db()
    db = client[MONGODB_NAME]
    collection = db.messages

    rows = collection.find()
    row_list = []
    async for row in rows:
        row_list.append(MessageInDB(**row))
    
    return row_list

def get_message_list(message_list):
    res = []
    for message in message_list:
        res.append(message.message)
    
    return res

def get_elastic_list(elastic_list):
    titles, urls = [], []
    cnt = 0
    for elastic in elastic_list:
        titles.append(elastic['_source']['title'])
        urls.append(elastic['_source']['url'])
        cnt += 1

        if cnt == 3:
            break
    
    return titles, urls

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=30001)