
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import logging


# --- Python logger ---
logging.basicConfig(
    level=logging.INFO, # Adjust to logging.DEBUG for more verbose logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AssessmentAnalysis:
    def __init__(self, data: list[dict], attendance_data: dict):
        self.data = data
        self.attendance_data = attendance_data

    def isDataEmpty(self)-> bool:
        return not self.data


    def isAttendanceDataEmpty(self)->bool:
        return not self.attendance_data

    def dataSize(self) ->int:
        return len(self.data)

    def data_(self):
        return self.data
    

    """
        Get list of sorted scores
    """
    def get_dataset_(self) -> list:
        moving_averages = []
        if self.isDataEmpty():
            return None
        
        for row in self.data:
            score = row.get("score")
            max_score = row.get("max_score")
            norm = (float(score) / float(max_score)) * 100
            moving_averages.append(norm)
        return moving_averages

    def get_dataset_labels_(self) -> list:
        if self.isDataEmpty():
            return None
        moving_averages = []
        for row in self.data:
            assessment_title = row.get("assessment_title")
            moving_averages.append(assessment_title)
        return moving_averages
    
    """
        Get defaultdict values of assessments sorted by alpha_identifier
    """
    def get_dataset_assessment_(self) ->list:
        assessment_dict = defaultdict(list)
        if self.isDataEmpty():
            return None
        for row in self.data:
            assessment_name, score, max_score = row.get("assessment_title"), row.get("score"), row.get("max_score")
            alpha_identifier, session_date = row.get("alpha_identifier"), row.get("session_date")
            pre, mid, post = row.get("pre"), row.get("mid"), row.get("post")
            classification = {"pre": pre, "mid": mid, "post": post}
            norm = (float(score) / float(max_score)) * 100
            # find al least one true for pre, mid, post
            key, value = None, None
            for k, v in classification.items():
                if v is True:
                    key = k
                    value = norm
            if key is not None and value is not None:
                assessment_dict[alpha_identifier].append(
                    {"alpha_identifier": alpha_identifier, "name": assessment_name, 
                     "score": norm, f"{key}": norm, "session_date": session_date, 'key_type': f"{key}"})

        result = []
        ##  [ KEY: [{score..} , {mid ..}]]
        for key, value in assessment_dict.items():
            classification = { "pre": 0, "mid": 0, "post": 0 }
            for item in value:
                if item['key_type'] in classification:
                    classification[item['key_type']] = item['score']

            result.append({"alpha_identifier": key} | classification)
                
        return result

    """
        Get defaultdict values with subject_name as its key
    """
    def get_dataset_subjects_(self) ->defaultdict:
        subject_sort = defaultdict(list)
        if self.isDataEmpty():
            return None
        for row in self.data:
            score = row.get("score")
            max_score = row.get("max_score")
            subject = row.get("subject")
            norm = (float(score) / float(max_score)) * 100
            subject_sort[subject].append(norm)

        return subject_sort

    def assessment_moving_average_(self)->dict:
        if self.isDataEmpty():
            return None
        moving_averages = self.get_dataset_()
        if moving_averages is None:
            return None
    
        df = pd.Series(moving_averages)
        SMA = df.rolling(window=5).mean().fillna(0).values
        EMA = df.ewm(span=5).mean().values
        CMA = df.expanding().mean().values
        frame = {'SMA': SMA.tolist(), "EMA": EMA.tolist(), "CMA": CMA.tolist() }
        return frame
    

    def subject_moving_average_bias_(self) -> list:
        if self.isDataEmpty():
            return None
        moving_average = self.get_dataset_subjects_()
        subjects = dict()
        for key, value in moving_average.items():
            df = pd.Series(value)
            percent_change  = df.pct_change().fillna(0).mean()
            mean = df.mean()
            if key not in subjects:
                subjects[key] = {"percent_change": percent_change, "mean": mean} 

            if key in subjects and mean > subjects[key]['mean']:
                subjects[key] = {"percent_change": percent_change, "mean": mean} 

        return [ {"subject": key, **value} for key, value in subjects.items()]
    
        


    def assessment_moving_average_subject_(self)->list:
        if self.isDataEmpty():
            return None
        subject_sort = self.get_dataset_subjects_()

        subjectdf = defaultdict()
        for key, values in subject_sort.items():
            df = pd.Series(values)
            SMA = df.mean()
            subjectdf[f'SMA:{key}'] = SMA
    
        return [{key: v} for key, v in  dict(subjectdf).items() ]
    

    """
        Requires the questionnare column to exist otherwise return None
    """
    def assessment_analysis_lr_(self)->list:
        if self.isDataEmpty():
            return None
        if self.isAttendanceDataEmpty():
            return None
        linear_regression_model = None
        try:
            with open('./Models/linear_model.pkl', 'rb') as file:
                linear_regression_model = pickle.load(file)
        except OSError as e:
            logging.error("unable to load linear_model.pkl")
            return None
        
        attendance_ratio = float (self.attendance_data.get("present"))/ float(self.attendance_data.get("total_sessions") ) * 100
        predictions = []
        for row in self.data:
            sport_hours = row.get("sports_hours")
            tutor_sessions = row.get("tutor_sessions")
            study_hours = row.get("study_hours")
            score = row.get("score")
            assessment_title = row.get("title")
            max_score = row.get("max_score")
            norm = float(score) / float(max_score) * 100
            df = pd.DataFrame([{
                "Hours_Studied": float(study_hours),
                "Attendance": float(attendance_ratio),
                "Previous_Scores": float(norm),
                "Tutoring_Sessions": float(tutor_sessions),
                "Physical_Activity": float(sport_hours),
            }])
            prediction = linear_regression_model.predict(df)[0]
            predictions.append({"prediction": float(prediction), "actual": float(norm), "title": assessment_title})
        return predictions
    
    """
        Requires the questionnare column to exist otherwise returns None
    """
    def assessment_analysis_lr_subject_(self)->list:
        if self.isDataEmpty():
            return None
        if self.isAttendanceDataEmpty():
            return None
        linear_regression_model = None
        try:
            with open('./Models/linear_model.pkl', 'rb') as file:
                linear_regression_model = pickle.load(file)
        except OSError as e:
            logger.error("unable to load linear_model")
            return None
        attendance_ratio = float (self.attendance_data.get("present"))/ float(self.attendance_data.get("total_sessions") ) * 100
        subjectsort = defaultdict(list)
        
        for row in self.data:
            sport_hours = row.get("sports_hours")
            tutor_sessions = row.get("tutor_sessions")
            study_hours = row.get("study_hours")
            score = row.get("score")
            max_score = row.get("max_score")
            norm = float(score) / float(max_score) * 100
            subject = row.get("subject")
            df = pd.DataFrame([{
                "Hours_Studied": float(study_hours),
                "Attendance": float(attendance_ratio),
                "Previous_Scores": float(norm),
                "Tutoring_Sessions": float(tutor_sessions),
                "Physical_Activity": float(sport_hours),
            }])
            prediction = linear_regression_model.predict(df)[0]
            subjectsort[f'LR_{subject}'].append(float(prediction))
        
        return [{f'{row[0]}': row[1] } for row in subjectsort.items()]
    

