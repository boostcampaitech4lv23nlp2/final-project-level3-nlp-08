from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from bson import ObjectId

class User(BaseModel):
    username: str
    hashed_password: str
    salt: str
 
class UserInDB(User):
    _id: ObjectId
    date_created: datetime = Field(default_factory=datetime.utcnow)

class Message(BaseModel):
    user: UserInDB
    content: str = None

class MessageInDB(Message):
    _id: ObjectId
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Room(BaseModel):
    room_name: str
    members: Optional[List[UserInDB]] = []
    messages: Optional[List[MessageInDB]] = []
    last_pinged: datetime = Field(default_factory=datetime.utcnow)

class RoomInDB(Room):
    _id: ObjectId
    date_created: datetime = Field(default_factory=datetime.utcnow)