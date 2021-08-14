import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import pandas as pd
import time
import numpy as np

# load the model
def prepare_data(eeg_df):
  file_names = eeg_df['Unnamed: 0'].tolist()

  subject_ids = []
  chunk_ids = []
  for fn in file_names:
    subject_ids.append(fn.split('.')[-1])
    chunk_ids.append(fn.split('.')[0])
  subject_ids = list(set(subject_ids))
  assert len(subject_ids) == 500

  sub2ind = {}
  for ind, sub in enumerate(subject_ids):
    sub2ind[sub] = ind

  eeg_combined = np.zeros((500, int(178*23)))
  labels_combined = np.zeros(500)
  labels_chunks = np.zeros((500, 23))
  labels_dict = {}
  for i in range(len(eeg_df)):
    fn = eeg_df.iloc[i]['Unnamed: 0']
    subject_id = fn.split('.')[-1]
    subject_ind = sub2ind[subject_id]

    chunk_id = int(fn.split('.')[0].split('X')[-1])
    start_idx = (chunk_id - 1) * 178
    end_idx = start_idx + 178
    eeg_combined[subject_ind, start_idx:end_idx] = eeg_df.iloc[i].values[1:-1]

    if subject_id not in labels_dict:
      labels_dict[subject_id] = []
    labels_dict[subject_id].append(eeg_df.iloc[i].values[-1])

  for sub_id, labels in labels_dict.items():
    sub_ind = sub2ind[sub_id]
    is_seizure = int(np.any(np.array(labels) == 1))
    labels_combined[sub_ind] = is_seizure
    labels = np.array(labels)
    labels = np.where(labels>1, 0, labels)
    labels_chunks[sub_ind,:] = labels

  return eeg_combined, labels_combined, labels_chunks

lstm = load_model('lstm_model.h5')


st.title("Seizure Prediction Model")
st.header("Upload the EEG dataset")
file_upload = st.file_uploader("Upload EEG")

if file_upload is not None:
  eeg = pd.read_csv(file_upload)
  eeg_data, _, _ = prepare_data(eeg)
  input_data = eeg_data.reshape(-1, 23, 178).astype(np.float32)
  
  # give some time until model does prediction
  with st.spinner('wait for the prediction to complete...'):
    time.sleep(12)
  
  # TODO 5: Use predict method on loaded_cnn and get the prediction
  prediction = lstm.predict(input_data)
  prediction = np.array((prediction >= 0.5))
  output_data = pd.DataFrame(prediction, columns=["Predictions"])
  output_data["Seizure?"] = output_data["Predictions"].replace({True:"Seizure", False: "No Seizure"})
  # finally check the prediction and print out result for the user
  st.dataframe(output_data)
  # if prediction == 1:
  #   st.error('Seizure Detected')
  # else:
  #   st.success('No Seizure Detected')