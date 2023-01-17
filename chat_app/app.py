from fastapi import FastAPI, WebSocket, Request, Depends
from starlette.websockets import WebSocket, WebSocketDisconnect
from typing import List
from fastapi.templating import Jinja2Templates
from mongodb import close_mongo_connection, connect_to_mongo, get_nosql_db, AsyncIOMotorClient
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from collections import defaultdict
from config import MONGODB_NAME
from controllers import create_user, verify_password, insert_room, get_rooms, get_room
import pymongo
import logging

app = FastAPI()









templates = Jinja2Templates(directory='templates')

class Notifier:
    def __init__(self):
        self.connections: dict = defaultdict(dict)
        self.generator = self.get_notification_generator()

    async def get_notification_generator(self):
        while True:
            message = yield
            msg = message['message']
            room_name = message['room_name']
            await self._notify(msg, room_name)

    async def push(self, msg: str, room_name: str = None):
        message_body = {'message': msg, 'room_name': room_name}
        await self.generator.asend(message_body)

    async def connect(self, websocket: WebSocket, room_name: str):
        await websocket.accept()
        if self.connections[room_name] == {} or len(self.connections[room_name]) == 0:
            self.connections[room_name] = []
        self.connections[room_name].append(websocket)
        #self.connections[room_name] = [] if self.connections[room_name] is None else self.connections[room_name].append(websocket)
        

    def remove(self, websocket: WebSocket, romm_name: str):
        self.connections[room_name].remove(websocket)

    async def _notify(self, message: str, room_name: str):
        living_connections = []
        while len(self.connections[room_name]) > 0:
            # Looping like this is necessary in case a disconnection is handled
            # during await websocket.send_text(message)
            websocket = self.connections[room_name].pop()
            await websocket.send_text(message)
            living_connections.append(websocket)
        self.connections[room_name] = living_connections

notifier = Notifier()

@app.on_event('startup')
async def startup_event():
    await connect_to_mongo()
    client = await get_nosql_db()
    db = client[MONGODB_NAME]
    try:
        await db.create_collection('users')
        await db.create_collection('rooms')
        await db.create_collection('messages')
        user_collection = db.users
        room_collection = db.rooms
        await user_collection.create_index('username', name='username', unique=True)
        await room_collection.create_index('room_name', name='room_name', unique=True)
    except pymongo.errors.CollectionInvalid as e:
        logging.info(e)
        pass

    
@app.on_event('shutdown')
async def shutdown_event():
    await close_mongo_connection()

@app.get('/{room_name}')
async def get(request: Request, room_name):
    return templates.TemplateResponse('index.html', {'request': request, 'room_name': room_name})



@app.websocket('/ws/{room_name}')
async def websocket_endpoint(websocket: WebSocket, room_name):
    await notifier.connect(websocket, room_name)
    try:
        while True:
            data = await websocket.receive_text()
            #print()
            #await websocket.send_text(f'{data}')
            await notifier.push(f'{data}', room_name)
    except WebSocketDisconnect:
        notifier.remove(websocket, room_name)

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RoomCreateRequest(BaseModel):
    user_id: str
    room_name: str

@app.put('/register')
async def register_user(request: RegisterRequest, client: AsyncIOMotorClient = Depends(get_nosql_db)):
    try:
        db = client[MONGODB_NAME]
        collection = db.users
        res = await create_user(request, collection)
        return res
    except pymongo.errors.DuplicateKeyError as e:
        return {'error': e}

@app.put('/login')
async def login_user(request: RegisterRequest, client: AsyncIOMotorClient = Depends(get_nosql_db)):
    db = client[MONGODB_NAME]
    collection = db.users
    row = await collection.find_one({'username': request.username})
    if row:
        return {'verified': verify_password((request.password + row['salt']), row['hashed_password'])}
    else:
        return {'verified': False}

@app.put('/create_room')
async def creat_room(request: RoomCreateRequest, client: AsyncIOMotorClient = Depends(get_nosql_db)):
    db = client[MONGODB_NAME]
    collection = db.rooms
    res = await insert_room(request.user_id, request.room_name, collection)
    return res

@app.get('/rooms')
async def get_all_rooms(client: AsyncIOMotorClient = Depends(get_nosql_db)):
    rooms = await get_rooms()
    return rooms

@app.get('/room/{room_name}')
async def get_single_room(room_name):
    room = await get_room(room_name)
    return room
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")