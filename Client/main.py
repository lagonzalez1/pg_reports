import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import json
load_dotenv()

class Client:
    def __init__(self, body):
        try:
            self.payload = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            print("Unable to prase client body")
    
    def get_output_key(self) -> str:
        return str(self.payload.get("s3_output_key")) 
    
    def get_student_id(self):
        return self.payload.get("student_id")

    def get_semester_id(self):
        try:
            semester_id = self.payload.get("semester_id")
            return semester_id
        except SyntaxError as e:
            return None



