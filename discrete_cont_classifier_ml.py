from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC

def train_motion_classifier(emg_signals, label):
    ml_model = RFC()
    ml_model.fit(emg_signals, label)
    
def pred_motion_class(ml_model_trained, emg_signal):
    pred =  ml_model_trained.predict(emg_signal)
    return pred
