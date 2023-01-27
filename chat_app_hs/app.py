from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, Depends
from typing import List
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from mongodb import get_nosql_db, connect_to_mongo, close_mongo_connection
from starlette.staticfiles import StaticFiles
from Elastic import elastic
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_NAME
from retriever.utils import *

from datetime import datetime
import summary
import logging
import uvicorn

    
app = FastAPI()
es = elastic.ElasticObject('localhost:9200')
#app.mount("/assets", app=StaticFiles(directory="assets"), name='assets')

r_tokenizer = AutoTokenizer.from_pretrained("LeeHJ/colbert_blog")
r_model = AutoModel.from_pretrained("LeeHJ/colbert_blog")

templates = Jinja2Templates(directory='templates')

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
    
class SocketManager:
    def __init__(self):
        self.active_connections: List[(WebSocket, str)] = []
        self.update_time = datetime.now()

    async def connect(self, websocket: WebSocket, user: str):
        await websocket.accept()
        self.active_connections.append((websocket, user))

    def disconnect(self, websocket: WebSocket, user: str):
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
    db = client[MONGODB_NAME]
    collection = db.messages

    if sender:
        await manager.connect(websocket, sender)
        response = {
            "location": "chat",
            "sender": sender,
            "message": sender + " 님이 접속하셨습니다."
        }
        await manager.broadcast(response)

        try:
            while True:
                data = await websocket.receive_json()
                res = await stack_message(data, collection)
                messages = await get_messages()
                message_list = get_message_list(messages)
                context = ''

                
                if data['recommend'] == 'True':
                    elastic_list = es.search('final_data', data['message'])
                    sources = get_elastic_list(elastic_list)

                    recommend_message = "# " + data['message']
                    recommend_data = {'location': 'recommend', 'sender': 'Golden Retriever', 'message': recommend_message, 'source': sources}
                    await manager.broadcast(recommend_data)
                else:
                    await manager.broadcast(data)

                if get_message_list_token(message_list) >= 30 or (manager.check_recommend() and get_message_list_token(message_list)) >= 50:
                    context = '<s>' + messages[0].message
                    context = "</s> <s>".join(message_list)
                    context = context + '</s>'
                    summary_context = summary.summarize(context)

                    summary_message = "지금까지 한 대화를 요약해봤어!" + "<br>" + summary_context
                    summary_data = {'location': 'summary', 'sender': 'Golden Retriever', 'message': summary_message}
                    await manager.broadcast(summary_data)
                    
                    #elastic_list = es.search('final_data', summary_context)
                    #sources = get_elastic_list(elastic_list)
                    
                    outputs = es.search('final_data', summary_context)
                    contexts = [output['_source']['context'] for output in outputs]
                    urls = [output['_source']['url'] for output in outputs]
                    
                    batched_p_embs = []

                    #블로그 글 임베딩하기
                    with torch.no_grad():
                        print("Start passage embedding......")
                        p_embs = []
                        for step, p in enumerate(tqdm(contexts)):
                            p = tokenize_colbert(p, tokenizer, corpus="doc").to("cuda")
                            p_emb = model.doc(**p).to("cpu").numpy()
                            p_embs.append(p_emb)
                            if (step + 1) % 200 == 0: # TODO 배치 단위 부분.
                                batched_p_embs.append(p_embs)
                                p_embs = []
                        batched_p_embs.append(p_embs)
                    with torch.no_grad():
                        model.eval()
                        q_seqs_val = tokenize(query, tokenizer).to("cuda")
                        q_emb = model.query(**q_seqs_val).to("cpu")
                    dot_prod_scores = get_score(q_emb, batched_p_embs, eval=True)
                    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
                    
                    sources = rank[:3] #TODO
                    recommend_message = "이것도 읽어봐라 멍멍!!"
                    recommend_data = {'location': 'recommend', 'sender': 'Golden Retriever', 'message': recommend_message, 'source': sources}
                    await manager.broadcast(recommend_data)

                    collection.delete_many({})
                                 
        except WebSocketDisconnect:
            manager.disconnect(websocket, sender)
            response['message'] = sender + "님이 접속을 종료했습니다."
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