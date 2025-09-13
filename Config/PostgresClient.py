import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import OperationalError, ProgrammingError, Error
from dotenv import load_dotenv
import logging

# --- 1. Set up basic logging to stdout ---
logging.basicConfig(
    level=logging.INFO, # Adjust to logging.DEBUG for more verbose logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

class PostgresClient:
    def __init__(self):
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Internal method to handle the database connection and logging."""
        try:
            logger.info("Attempting to connect to PostgreSQL database.")
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_URL"),
                port=os.getenv("POSTGRES_PORT"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                dbname=os.getenv("POSTGRES_DB_NAME")
            )
            self.conn.autocommit = True
            logger.info("Successfully connected to PostgreSQL database.")
        except OperationalError as e:
            # This handles connection-related errors
            logger.error("Failed to connect to PostgreSQL database.")
            logger.exception(e)
            raise RuntimeError("Database connection failed") from e
        except Exception as e:
            logger.exception("An unexpected error occurred during database connection.")
            raise RuntimeError("Database connection failed") from e

    def _get_cursor(self, cursor_factory=None):
        """Internal helper to get a cursor and handle potential connection issues."""
        if not self.conn or self.conn.closed:
            logger.warning("Database connection is closed. Attempting to reconnect...")
            self._connect()
        return self.conn.cursor(cursor_factory=cursor_factory)

    def fetch_one(self, query, params=None):
        try:
            with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                logger.debug(f"Executed query: {query} with params: {params}")
                return cursor.fetchone()
        except (OperationalError, ProgrammingError) as e:
            logger.error(f"Failed to execute query: {query}")
            logger.exception(e)
            raise RuntimeError("Database query failed") from e

    def fetch_all(self, query, params=None):
        try:
            with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                logger.debug(f"Executed query: {query} with params: {params}")
                return cursor.fetchall()
        except (OperationalError, ProgrammingError) as e:
            logger.error(f"Failed to execute query: {query}")
            logger.exception(e)
            raise RuntimeError("Database query failed") from e

    def execute(self, query, params=None):
        try:
            with self._get_cursor() as cursor:
                cursor.execute(query, params)
                logger.info(f"Executed command: {query} with params: {params}")
        except (OperationalError, ProgrammingError) as e:
            logger.error(f"Failed to execute command: {query}")
            logger.exception(e)
            raise RuntimeError("Database command failed") from e
    

    def get_all_student_assessments(self, student_id, semester_id: None):
        sql = [
            """
                SELECT 
                    ss.session_date,
                    ast.score,
                    asmt.max_score,
                    asmt.subject_id,
                    sj.title AS subject,
                    asmt.pre,
                    asmt.post,
                    asmt.mid,
                    asmt.alpha_identifier,
                    asmt.title AS assessment_title
                FROM stu_tracker.Assessments_students ast
                LEFT JOIN stu_tracker.Sessions ss ON
                    ss.id = ast.session_id
                LEFT JOIN stu_tracker.Assessments asmt ON
                    asmt.id = ast.assessment_id
                LEFT JOIN stu_tracker.Subjects sj ON
                    sj.id = asmt.subject_id
                WHERE ast.student_id = %s
            """
        ]
        params = [student_id]
        if semester_id is not None:
            sql.append("AND ast.semester_id = %s")
            params.append(semester_id)

        sql.append("ORDER BY ss.session_date DESC;")
        query = " ".join(sql) 
        data_cursor = self.fetch_all(query, params)
        if len(data_cursor) == 0:
            return None
        data = [dict(row) for row in data_cursor]
        return data
    
    def get_student_prior_assessments(self, student_id, semester_id: None):
        sql = [
            """
                SELECT 
                    ss.session_date,
                    ast.score,
                    asmt.max_score,
                    asmt.subject_id,
                    sj.title AS subject,
                    asmt.pre,
                    asmt.post,
                    asmt.mid 
                FROM stu_tracker.Assessments_students ast
                LEFT JOIN stu_tracker.Sessions ss ON
                    ss.id = ast.session_id
                LEFT JOIN stu_tracker.Assessments asmt ON
                    asmt.id = ast.assessment_id
                LEFT JOIN stu_tracker.Subjects sj ON
                    sj.id = asmt.subject_id
                WHERE ast.student_id = %s
            """
        ]
        params = [student_id]
        if semester_id is not None:
            sql.append("AND ast.semester_id = %s")
            params.append(semester_id)

        sql.append("AND ast.questionnaire_id IS NULL")
        sql.append("ORDER BY ss.session_date DESC;")
        query = " ".join(sql) 
        data_cursor = self.fetch_all(query, params)
        if len(data_cursor) == 0:
            return None
        data = [dict(row) for row in data_cursor]
        return data

    
    def get_student_prior_assessments_guestionnaire(self, student_id, semester_id: None):
        sql = [
            """
                SELECT 
                    ss.session_date,
                    ast.score,
                    asmt.max_score,
                    asmt.subject_id,
                    asmt.title,
                    paq.sleep_hours,
                    paq.effort_score,
                    paq.tutor_sessions,
                    paq.sports_hours,
                    paq.peer_influence,
                    paq.study_hours,
                    paq.id AS questionnaire_id,
                    sj.title AS subject
                FROM stu_tracker.Assessments_students ast
                LEFT JOIN stu_tracker.Sessions ss ON
                    ss.id = ast.session_id
                LEFT JOIN stu_tracker.Assessments asmt ON
                    asmt.id = ast.assessment_id
                LEFT JOIN stu_tracker.Pre_assessment_questionnaire paq ON
                    paq.id = ast.questionnaire_id
                LEFT JOIN stu_tracker.Subjects sj ON
                    sj.id = asmt.subject_id
                WHERE ast.student_id = %s
            """
        ]
        params = [student_id]
        if semester_id is not None:
            sql.append("AND ast.semester_id = %s")
            params.append(semester_id)

        sql.append("AND ast.questionnaire_id IS NOT NULL")
        sql.append("ORDER BY ss.session_date DESC;")
        query = " ".join(sql) 
        data_cursor = self.fetch_all(query, params)
        if len(data_cursor) == 0:
            return None
        data = [dict(row) for row in data_cursor]
        return data

    def get_subject_(self):
        return None


    def get_student_questionnaire(self, params):
        query = (
        "SELECT" 
        "q.subject_id AS subject_id,"
        "q.created_at AS date,"
        "q.study_hours, " 
        "q.sleep_hours, " 
        "q.effort_score, " 
        "q.tutor_sessions, " 
        "q.parental_help, " 
        "q.sports_hours, " 
        "q.peer_influence, " 
        "q.assessment_id "
        "FROM stu_tracker.Pre_assessment_questionnaire q"
        "LEFT JOIN stu_tracker.Assessments ast " 
        "ON ast.id = q.assessment_id"
        "WHERE student_id = %s")
        data_cursor = self.fetch_all(query, params)
        return [dict(row) for row in data_cursor]
    

    def get_student_attendance_semester_filter(self, params):
        query = (
            "SELECT "
            "COUNT(*) AS total_sessions," 
            "SUM( CASE WHEN NOT ss.absent THEN 1 ELSE 0 END) AS present,"
            "SUM( CASE WHEN ss.absent THEN 1 ELSE 0 END) AS absent" 
            "FROM stu_tracker.Session_students ss" \
            "LEFT JOIN stu_tracker.Sessions st" \
            "ON st.id = ss.session_id" \
            "WHERE ss.student_id = %s AND st.semester_id = %s"
            "GROUP BY ss.student_id;"
        )
        return self.fetch_one(query, params)

    ## Can filter by semester_id
    def get_student_attendance(self, student_id, semester_id: None):

        sql = [
            """
                SELECT
                COUNT(*) AS total_sessions,
                SUM( CASE WHEN NOT ss.absent THEN 1 ELSE 0 END) AS present,
                SUM( CASE WHEN ss.absent THEN 1 ELSE 0 END) as absent
                FROM stu_tracker.Session_students ss
            """
        ]
        params = [student_id]
        if semester_id is not None:
            sql.append("LEFT JOIN stu_tracker.Sessions st ON st.id = ss.session_id")
            sql.append("WHERE ss.student_id = %s AND st.semester_id = %s")
            sql.append("GROUP BY ss.student_id;")
            params.append(semester_id)
            
        else:
            sql.append("WHERE ss.student_id = %s")
            sql.append("GROUP BY ss.student_id;")        
        
        query = " ".join(sql)
        cursor =  self.fetch_one(query, params)
        if cursor is None:
            return None
        else:
            return dict(cursor)

    def update_event_queue(self, params):
        query = [
            """
                UPDATE stu_tracker.Student_report 
                SET status = %s WHERE s3_output_key = %s
            """
        ]
        
        q = " ".join(query) 
        self.execute(q, params)
    
    def get_subject_data(self, params):
        subject_query = "SELECT title, description FROM stu_tracker.Subjects WHERE organization_id = %s AND id = %s"
        return self.fetch_one(subject_query, params)
            
    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("PostgreSQL connection closed.")