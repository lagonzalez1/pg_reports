import pickle
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, # Adjust to logging.DEBUG for more verbose logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class DisabilityAnalysis:
    def __init__(self, assessment_data: list[dict], attendance_data: dict):
        self.assessment_data = assessment_data
        self.attendance_data = attendance_data
    
    def isAssessmentDataEmpty(self) ->bool:
        return not self.assessment_data
    
    def isAttendanceDataEmpty(self)->bool:
        return not self.attendance_data

    def assessment_data_values_(self) ->list:
        values = []
        for row in self.assessment_data:
            score = row.get('score')
            max_score = row.get('max_score')
            norm = float(score/max_score) * 100 
            values.append(norm)
        return values


    """
        Package list of classifications into descriptive dict
    """
    def prediction_dict(self, predictions_: list) -> dict:
        pred = None
        if len(predictions_) <= 3:
            return {
                "classification": int(0),
                "notes": 'Not enough data to make prognosis.',
                "confidence": float(100),
                "data": self.assessment_data_values_()
            }
        positive = predictions_.count(1)
        negative = predictions_.count(0)
        if positive == negative:
            pred = {
                "classification": int(0),
                "notes": 'Split prediction, unsure of prognosis.',
                "confidence": float(50),
                "data": self.assessment_data_values_()
            }
        elif positive > negative:
            confidence = float(positive / len(predictions_)) * 100
            pred = {
                "classification": int(1),
                "notes": 'Postitive classification, based on previous assessment scores and questionnaires',
                "confidence": float(confidence),
                "data": self.assessment_data_values_()
            }
        else:
            confidence = float(negative / len(predictions_)) * 100
            pred = {
                "classification": int(1),
                "notes": 'Negative classification, based on previous assessment scores and questionnaires',
                "confidence": float(confidence),
                "data": self.assessment_data_values_()
            }
        return pred
    

    def get_attendance_ratio(self)->float:
        return float(self.attendance_data.get('present')) /float(self.attendance_data.get('total_sessions')) * 100 

    def student_analysis_(self) ->dict:
        if self.isAssessmentDataEmpty():
            return None
        if self.isAttendanceDataEmpty():
            return None
        disability_model = None
        try:
            with open("./Models/logistic_model.pkl", 'rb') as file:
                disability_model = pickle.load(file)
        except OSError as e:
            logging.info("unable to load logistic_model.pkl")
            return None
        
        attendance_ratio = self.get_attendance_ratio()
        predictions_ = []
        for i in range(1, len(self.assessment_data)):
            prev_row = self.assessment_data[i-1]
            row = self.assessment_data[i]
            exam_score = float(row.get("score") / row.get("max_score")) * 100
            prev_score = float(prev_row.get("score") / row.get("max_score")) * 100
            tutor_sessions = float(row.get("tutor_sessions"))
            df = pd.DataFrame([{
                "Attendance": float(attendance_ratio),
                "Previous_Scores": float(prev_score),
                "Exam_Score": float(exam_score),
                "Tutoring_Sessions": float(tutor_sessions)
            }])
            prediction = disability_model.predict(df)[0]
            predictions_.append(prediction)
        
        return self.prediction_dict(predictions_)

