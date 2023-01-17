import uuid
import hashlib
from models import User, UserInDB, Room, RoomInDB, Message, MessageInDB
from mongodb import get_nosql_db
from config import MONGODB_NAME
from bson import ObjectId


async def create_user(request, collection):
    salt = uuid.uuid4().hex
    hashed_password = hashlib.sha512(request.password.encode('utf-8') + salt.encode('utf-8')).hexdigest()

    user = {}
    user['username'] = request.username
    user['salt'] = salt
    user['hashed_password'] = hashed_password

    dbuser = UserInDB(**user)
    response = await collection.insert_one(dbuser.dict())
    return {'id_inserted': str(response.inserted_id)}


async def get_user(id) -> User:
    client = await get_nosql_db()
    db = client[MONGODB_NAME]
    collection = db.users
    row = await collection.find_one({'_id': ObjectId(id)})
    if row is not None:
        return UserInDB(**row)
    else:
        return None

def verify_password(plain_password_w_salt, hashed_password):
    checked_password = hashlib.sha512(plain_password_w_salt.encode('utf-8')).hexdigest()
    return checked_password == hashed_password

async def insert_room(user_id, room_name, collection):
    room = {}
    room['room_name'] = room_name
    user = await get_user(user_id)
    room['members'] = [user] if user is not None else []

    dbroom = RoomInDB(**room)
    response = await collection.insert_one(dbroom.dict())
    return {'id_inserted': str(response.inserted_id)}

async def get_rooms():
    client = await get_nosql_db()
    db = client[MONGODB_NAME]
    collection = db.rooms
    rows = collection.find()
    row_list = []
    async for row in rows:
        row_list.append(RoomInDB(**row))

    return row_list

async def get_room(room_name) -> RoomInDB:
    client = await get_nosql_db()
    db = client[MONGODB_NAME]
    collection = db.rooms
    row = await collection.find_one({'room_name': room_name})
    if row is not None:
        return RoomInDB(**row)
    else:
        return None