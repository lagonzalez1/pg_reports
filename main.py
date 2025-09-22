import os
from Config.RabbitMQ import RabbitMQ
from Config.PostgresClient import PostgresClient
from Assessment_analysis.main import AssessmentAnalysis
from Disability_analysis.main import DisabilityAnalysis
from S3.main import S3Instance
from Client.main import Client
from dotenv import load_dotenv
import time
import json
import logging

# --- 1. Set up basic logging to stdout ---
logging.basicConfig(
    level=logging.INFO, # Adjust to logging.DEBUG for more verbose logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


EXCHANGE     = os.getenv("EXCHANGE")
QUEUE        = os.getenv("QUEUE")
ROUTING_KEY  = os.getenv("ROUTING_KEY")
RABBIT_LOCAL  = os.getenv("RABBIT_LOCAL")
PREFETCH_COUNT = 1
EXCHANGE_TYPE = "direct"
ERROR = "ERROR"
DONE = "DONE"

def create_callback(db):
    def on_message_test(channel, method, properties, body):
        client = Client(body)
        assessment_data_all = db.get_all_student_assessments(client.get_student_id(), client.get_semester_id())
        assessment_data_w_q = db.get_student_prior_assessments_guestionnaire(client.get_student_id(), client.get_semester_id())
        attendance_data = db.get_student_attendance(client.get_student_id(), client.get_semester_id())

        da = DisabilityAnalysis(assessment_data_w_q, attendance_data)
        an = AssessmentAnalysis(assessment_data_all, attendance_data)
        anq = AssessmentAnalysis(assessment_data_w_q, attendance_data)
        df = {
            "generated_at": time.time(),
            "all_scores": {
                "scores": an.assessment_moving_average_(),
                "data": an.get_dataset_(),
                "labels": an.get_dataset_labels_()
            },
            "subject_bias": an.subject_moving_average_bias_(),
            "assessment_comparison" : an.get_dataset_assessment_(),
            "learning_disability": da.student_analysis_(), 
            "learning_disability_linear_regression": {
                "scores_linear_regression": anq.assessment_analysis_lr_(),
            }
        }
        try:
            js = json.dumps(df)
            s3 = S3Instance("tracker-student-reports")
            ### utf-8 will make it convertable on the frontend Parsable
            s3.put_object(client.get_output_key(), js.encode('utf-8'))
            db.update_event_queue((DONE, client.get_output_key()))
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except TypeError as e:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            db.update_event_queue((ERROR, client.get_output_key()))
            
    return on_message_test

        

def main():
    db = PostgresClient()
    mq = RabbitMQ(PREFETCH_COUNT, EXCHANGE, QUEUE, ROUTING_KEY, EXCHANGE_TYPE)
    callback = create_callback(db)
    mq.set_callback(callback)
    channel = mq.get_channel()
    connection = mq.get_connection()
    logging.info(f"[*] Waiting for message in {QUEUE}. ")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        channel.close()
        connection.close()
        db.close()
    
if __name__ == "__main__":
    main()
