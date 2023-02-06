from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, Depends
from typing import List
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from mongodb import get_nosql_db, connect_to_mongo, close_mongo_connection
from starlette.staticfiles import StaticFiles
from datetime import datetime
from bson import ObjectId
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient
from src.elastic.elastic import ElasticObject
from urllib import parse
from config import MONGODB_NAME
import requests
import pprint
from datetime import timedelta
elastic_connector = ElasticObject("localhost:9200")    
app = FastAPI()

app.mount("/assets", app=StaticFiles(directory="assets"), name='assets')

templates = Jinja2Templates(directory='./templates')


@app.on_event('startup')
async def startup_event():
    if not elastic_connector.client.indices.exists(index='chat-history'):
        elastic_connector.create_index("chat-history", setting_path="./src/elastic/history_settings.json")

    await connect_to_mongo()
    client = await get_nosql_db()
    db = client[MONGODB_NAME]

    try:
        message_collection = db.messages
    except pymongo.errors.CollectionInvalid as e:
        logging.info(e)
        pass           

async def load_chat():
    try:
        body = {
            "size": 1000,
            "query": {
                "match_all": {}
            },
            "sort": [
                {
                    "date": {
                        "order": "asc"
                    }
                }
            ]
        }
        resp = elastic_connector.client.search(index="chat-history", body=body)
        if resp['hits']['hits']:
            for res in resp['hits']['hits']:
                await manager.broadcast(res['_source'])
    except:
        pass
    


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse('home.html', {"request": request})

@app.get("/chat", response_class=HTMLResponse)
def read_chat(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@app.get("/api/current_user")
def get_user(request: Request):
    return parse.unquote(request.cookies.get("X-Authorization"))

class RegisterValidator(BaseModel):
    username: str
    
    class Config:
        orm_mode = True
        
@app.post("/api/register")
def register_user(user: RegisterValidator, response: Response):
    response.set_cookie(key="X-Authorization", value=parse.quote(user.username), httponly=True)
    
    

async def summary_retrieve(summary):
    
    _, outputs = elastic_connector.search(index_name="naver_docs", question=summary['answer'], topk=5)
    
    return outputs

    
class SocketManager:
    def __init__(self):
        self.active_connections: List[(WebSocket, str)] = []
        self.update_time = datetime.now()

    async def connect(self, websocket: WebSocket, user: str):
        await websocket.accept()
        self.active_connections.append((websocket, user))

    def disconnect(self, websocket: WebSocket, user: str):
        print(self.active_connections)
        if len(self.active_connections) == 1:
            try:
                elastic_connector.client.indices.delete(index='chat-history')
            except:
                None
        self.active_connections.remove((websocket, user))

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            await connection[0].send_json(data)
            
    def check_recommend(self):
        now_time = datetime.now()
        if ((now_time - self.update_time).seconds / 60) > 5:
            self.update_time = now_time
            return True
        return False
        
manager = SocketManager()

@app.websocket("/api/chat")
async def chat(websocket: WebSocket, client: AsyncIOMotorClient = Depends(get_nosql_db)):
    sender = websocket.cookies.get("X-Authorization")
    sender = parse.unquote(sender)
    db = client[MONGODB_NAME]
    collection = db.messages
    if sender:
        await manager.connect(websocket, sender)
        response = {
            "location": "chat",
            "sender": sender,
            "message": sender + "님이 접속하셨습니다."
        }

        await manager.broadcast(response)
        await load_chat()
        try:
            while True:
                data = await websocket.receive_json()
                res = await stack_message(data, collection)
                messages = await get_messages()
                message_list = get_message_list(messages)
                context = ''
                await manager.broadcast(data)
                
                if (get_message_list_token(message_list) > 30 or (manager.check_recommend() and get_message_list_token(message_list)) >= 50) and check_speaker_change(messages):
                    context = '<s>' + messages[0].message
                    context = '</s> <s>'.join(message_list)
                    context = context + '</s>'

                    summary_output = requests.post("http://localhost:8502", json={"text": context}).json()
                    outputs = requests.post("http://localhost:8503", json={"text": summary_output['answer']}).json()

                    current_time = (datetime.now() - timedelta(hours=3)).strftime('%Y/%m/%d %H:%M:%S')
                    outputs['answer']['date'] = current_time
                    elastic_connector.client.index(index='chat-history',  body=outputs['answer'])

                    await manager.broadcast(outputs['answer'])

                    collection.delete_many({})
                        
        except WebSocketDisconnect:
            manager.disconnect(websocket, sender)
            response['message'] = sender + "님이 떠나셨습니다."
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

def check_speaker_change(messages):
    if messages[-1].username != messages[-2].username:
        return True
    else:
        return False

def get_message_list(messages):
    res = []
    for message in messages:
        res.append(message.message)
    
    return res

def get_message_list_token(message_list):
    cnt = 0
    for message in message_list:
        cnt += len(message.split(' '))
    
    return cnt

def get_elastic_list(elastic_list):
    sources = []
    cnt = 0
    for elastic in elastic_list:
        source = {'url': elastic['_source']['url'], 'title': elastic['_source']['title']}
        sources.append(source)
        cnt += 1

        if cnt == 3:
            break
    
    return sources

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30001)