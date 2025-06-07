import cv2
import mediapipe as mp
import numpy as np
import time
import os
import torch
from lstm.lstm_architecture import LSTMModel
from cnn.cnn_architecture import CNN
from seq2seq.encoder_architecture import Encoder
from seq2seq.decoder_architecture import Decoder
from seq2seq.predict import evaluate

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#lstm_model = LSTMModel()
LSTM_MODEL = torch.load('lstm/lstm_temp.pth', map_location=torch.device('cpu'), weights_only=False)
LSTM_MODEL.eval()  # Set to evaluation mode

CNN_MODEL = CNN()
CNN_MODEL.load_state_dict(torch.load('cnn/cnn.pth', map_location=torch.device('cpu'), weights_only=False))
CNN_MODEL.eval()  # Set to evaluation mode

ENCODER = Encoder()
ENCODER.load_state_dict(torch.load('seq2seq/encoder.pth', map_location=torch.device('cpu'), weights_only=False))

DECODER = Decoder()
DECODER.load_state_dict(torch.load('seq2seq/decoder.pth', map_location=torch.device('cpu'), weights_only=False))

# Define sign classes
WORDS_LABEL = ["hello","meet", "my","name","nice", "please","sit", "yes", "you"] 

ALPHABETS_LABEL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y']

MODE = 1  # 1 for LSTM, 0 for CNN

def extract_landmarks(results):
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extract hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenate all landmarks into a feature vector
    if(MODE==1): return np.concatenate([pose, lh, rh]) #LSTM
    return np.concatenate([lh, rh]) #CNN

def draw_styled_landmarks(image, results):
    # # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

# For frame collection and sequence processing
SEQUENCE_LENGTH = 50  #FRAME_LENGTH
SEQUENCE = []
THRESHOLD = 0.8  # Confidence threshold

cap = cv2.VideoCapture(0)

# For FPS calculation
PREV_TIME = 0
CURR_TIME = 0

# Initialize variables for alphabet tracking
ALPHABET_TOKEN_DICT = {}  # Dictionary to track predicted alphabets
WORD_TOKEN_DICT = {}  # Dictionary to track predicted words
SENTENCE = ""    # Sentence to store results
LAST_DETECTED_TIME = time.time()  # Time when hands were last detected
HAND_PRESENT = False  # Flag to track if hands are present
SENTENCE_RESET_TIME = 0
SENTENCE_PROCESSED = False 

try:
    while cap.isOpened():
        #changing mode by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            MODE = 1 if MODE == 0 else 0
            print(f"Mode changed to: {'LSTM' if MODE == 1 else 'CNN'}")
            ALPHABET_TOKEN_DICT = {}
            WORD_TOKEN_DICT = {}
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Calculate FPS
        CURR_TIME = time.time()
        fps = 1 / (CURR_TIME - PREV_TIME) if PREV_TIME > 0 else 0
        PREV_TIME = CURR_TIME
        
        # Process the frame
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Display FPS
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display mode
        cv2.putText(image, f'Mode: {"LSTM" if MODE == 1 else "CNN"}', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Extract landmarks for prediction
        if results.left_hand_landmarks or results.right_hand_landmarks:
            frame_features = extract_landmarks(results)
            HAND_PRESENT = True
            LAST_DETECTED_TIME = time.time()
            word=""
            SENTENCE_PROCESSED = False 

            
            if MODE == 1:
                
                SEQUENCE.append(frame_features)
                
                # Keep only the most recent frames
                if len(SEQUENCE) > SEQUENCE_LENGTH:
                    SEQUENCE = SEQUENCE[-SEQUENCE_LENGTH:]
                    
                # If we have enough frames, make a prediction
                if len(SEQUENCE) == SEQUENCE_LENGTH:
                    # Prepare the input tensor - explicitly on CPU
                    input_tensor = torch.tensor(np.array([SEQUENCE]), dtype=torch.float32)
                    
                    # Get the model prediction
                    with torch.no_grad():
                        output = LSTM_MODEL(input_tensor)
                        
                        
                    # Get the predicted class and confidence
                    confidence, predicted_class_idx = torch.max(output, dim=1)
                    predicted_class = WORDS_LABEL[predicted_class_idx.item()]
                    
                    # Display prediction if confidence is above threshold
                    if confidence.item() > THRESHOLD:
                        #add the predicted class and count to the token dictionary
                        word=predicted_class
                        if predicted_class in WORD_TOKEN_DICT:
                            WORD_TOKEN_DICT[predicted_class] += 1
                        else:
                            WORD_TOKEN_DICT[predicted_class] = 1
                        cv2.putText(image, f'Sign: {predicted_class}', (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, f'Confidence: {confidence.item():.2f}', (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            else:
                token = {}
                input_tensor = torch.tensor(np.array([frame_features]), dtype=torch.float32)
                with torch.no_grad():
                    output = CNN_MODEL(input_tensor)
                # Get the predicted class and confidence
                confidence, predicted_class_idx = torch.max(output, dim=1)
                predicted_class = ALPHABETS_LABEL[predicted_class_idx.item()]
                # Display prediction if confidence is above threshold
                if confidence.item() > THRESHOLD:
                    # Add the predicted class to the token dictionary
                    if predicted_class in ALPHABET_TOKEN_DICT:
                        ALPHABET_TOKEN_DICT[predicted_class] += 1
                    else:
                        ALPHABET_TOKEN_DICT[predicted_class] = 1
                    cv2.putText(image, f'Sign: {predicted_class}', (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Confidence: {confidence.item():.2f}', (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Display a message
                    cv2.putText(image, "No sign detected", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        else:
            SEQUENCE=[]
            if HAND_PRESENT:  # If hands were previously detected but not anymore
                HAND_PRESENT = False
                # In CNN mode, add most frequent alphabet to sentence when hands disappear
                if MODE == 0 and ALPHABET_TOKEN_DICT:  # Only if we have collected some predictions
                    most_frequent_alphabet = max(ALPHABET_TOKEN_DICT, key=ALPHABET_TOKEN_DICT.get) if ALPHABET_TOKEN_DICT else ""
                    if most_frequent_alphabet:  # Only if we have collected some predictions
                        SENTENCE += most_frequent_alphabet
                    ALPHABET_TOKEN_DICT = {}  # Reset the token dictionary
                else: 
                    most_frequent_word = max(WORD_TOKEN_DICT, key=WORD_TOKEN_DICT.get) if WORD_TOKEN_DICT else ""
                    if most_frequent_word:  # Only if we have collected some predictions
                        SENTENCE += most_frequent_word + " "
                    WORD_TOKEN_DICT = {}  # Reset the token dictionary
            
            # Check if no hand sign is detected for 5 seconds
            if time.time() - LAST_DETECTED_TIME > 6.5 and SENTENCE and not SENTENCE_PROCESSED:
                SENTENCE = evaluate(ENCODER, DECODER, SENTENCE)
                SENTENCE_PROCESSED = True
                print(f"Final sentence: {SENTENCE}")
                SENTENCE_RESET_TIME = time.time() + 3.5

            if SENTENCE_RESET_TIME > 0 and time.time() >= SENTENCE_RESET_TIME:
                SENTENCE = ""
                SENTENCE_RESET_TIME = 0    
                
            
            # Display a message
            cv2.putText(image, "No hands detected", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Always display the current sentence
        cv2.putText(image, f'Sentence:{SENTENCE}', (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Recognition', image)
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Make sure to release resources
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Destroy all windows
    holistic.close()  # Close the holistic model
    print("Application closed successfully")