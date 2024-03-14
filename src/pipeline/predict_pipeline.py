import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = 'artifact\model.pkl'
            preprocessor_path = 'artifact\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e :
            raise CustomException(e,sys)

class CustomData:
    '''
    Responsible for mapping all the data points we receive in the front end to the back end.
    '''
    def __init__(self,
                 Gender,
                 Married,
                 Dependents,
                 Education,
                 Self_Employed,
                 Property_Area,
                 ApplicantIncome, 
                 CoapplicantIncome,
                 LoanAmount, 
                 Loan_Amount_Term,
                 Credit_History):
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.Property_Area = Property_Area
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History

    def get_data_as_data_frame(self):
        try:
            custom_data_input_disk = {
                'Gender' : [self.Gender],
                'Married' : [self.Married],
                'Dependents' : [self.Dependents],
                'Education' : [self.Education],
                'Self_Employed' : [self.Self_Employed],
                'Property_Area' : [self.Property_Area],
                'ApplicantIncome' : [self.ApplicantIncome], 
                'CoapplicantIncome' : [self.CoapplicantIncome],
                'LoanAmount' : [self.LoanAmount],
                'Loan_Amount_Term' : [self.Loan_Amount_Term],
                'Credit_History' : [self.Credit_History]
            }

            return pd.DataFrame(custom_data_input_disk)
        
        except Exception as e:
            raise CustomException(e,sys)