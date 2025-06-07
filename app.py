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
lstm_model = torch.load('lstm/lstm_temp.pth', map_location=torch.device('cpu'), weights_only=False)
lstm_model.eval()  # Set to evaluation mode

cnn_model = CNN()
cnn_model.load_state_dict(torch.load('cnn/cnn.pth', map_location=torch.device('cpu'), weights_only=False))
cnn_model.eval()  # Set to evaluation mode

encoder = Encoder()
encoder.load_state_dict(torch.load('seq2seq/encoder.pth', map_location=torch.device('cpu'), weights_only=False))

decoder = Decoder()
decoder.load_state_dict(torch.load('seq2seq/decoder.pth', map_location=torch.device('cpu'), weights_only=False))

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
sequence_length = 50
sequence = []
predictions = []
threshold = 0.8  # Confidence threshold

cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0
curr_time = 0

# Initialize variables for alphabet tracking
alphabet_token_dict = {}  # Dictionary to track predicted alphabets
word_token_dict = {}  # Dictionary to track predicted words
sentence = ""    # Sentence to store results
last_detected_time = time.time()  # Time when hands were last detected
hand_present = False  # Flag to track if hands are present
sentence_reset_time = 0
sentence_processed = False 

try:
    while cap.isOpened():
        #changing mode by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            MODE = 1 if MODE == 0 else 0
            print(f"Mode changed to: {'LSTM' if MODE == 1 else 'CNN'}")
            alphabet_token_dict = {}
            word_token_dict = {}
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
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
            hand_present = True
            last_detected_time = time.time()
            word=""
            sentence_processed = False 

            
            if MODE == 1:
                
                sequence.append(frame_features)
                
                # Keep only the most recent frames
                if len(sequence) > sequence_length:
                    sequence = sequence[-sequence_length:]
                    
                # If we have enough frames, make a prediction
                if len(sequence) == sequence_length:
                    # Prepare the input tensor - explicitly on CPU
                    input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32)
                    
                    # Get the model prediction
                    with torch.no_grad():
                        output = lstm_model(input_tensor)
                        
                        
                    # Get the predicted class and confidence
                    confidence, predicted_class_idx = torch.max(output, dim=1)
                    predicted_class = WORDS_LABEL[predicted_class_idx.item()]
                    
                    # Display prediction if confidence is above threshold
                    if confidence.item() > threshold:
                        #add the predicted class and count to the token dictionary
                        word=predicted_class
                        if predicted_class in word_token_dict:
                            word_token_dict[predicted_class] += 1
                        else:
                            word_token_dict[predicted_class] = 1
                        cv2.putText(image, f'Sign: {predicted_class}', (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, f'Confidence: {confidence.item():.2f}', (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            else:
                token = {}
                input_tensor = torch.tensor(np.array([frame_features]), dtype=torch.float32)
                with torch.no_grad():
                    output = cnn_model(input_tensor)
                # Get the predicted class and confidence
                confidence, predicted_class_idx = torch.max(output, dim=1)
                predicted_class = ALPHABETS_LABEL[predicted_class_idx.item()]
                # Display prediction if confidence is above threshold
                if confidence.item() > threshold:
                    # Add the predicted class to the token dictionary
                    if predicted_class in alphabet_token_dict:
                        alphabet_token_dict[predicted_class] += 1
                    else:
                        alphabet_token_dict[predicted_class] = 1
                    cv2.putText(image, f'Sign: {predicted_class}', (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Confidence: {confidence.item():.2f}', (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Display a message
                    cv2.putText(image, "No sign detected", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        else:
            sequence=[]
            if hand_present:  # If hands were previously detected but not anymore
                hand_present = False
                # In CNN mode, add most frequent alphabet to sentence when hands disappear
                if MODE == 0 and alphabet_token_dict:  # Only if we have collected some predictions
                    most_frequent_alphabet = max(alphabet_token_dict, key=alphabet_token_dict.get) if alphabet_token_dict else ""
                    if most_frequent_alphabet:  # Only if we have collected some predictions
                        sentence += most_frequent_alphabet
                    alphabet_token_dict = {}  # Reset the token dictionary
                else: 
                    most_frequent_word = max(word_token_dict, key=word_token_dict.get) if word_token_dict else ""
                    if most_frequent_word:  # Only if we have collected some predictions
                        sentence += most_frequent_word + " "
                    word_token_dict = {}  # Reset the token dictionary
            
            # Check if no hand sign is detected for 5 seconds
            if time.time() - last_detected_time > 6.5 and sentence and not sentence_processed:
                sentence = evaluate(encoder, decoder, sentence)
                sentence_processed = True
                print(f"Final sentence: {sentence}")
                sentence_reset_time = time.time() + 3.5

            if sentence_reset_time > 0 and time.time() >= sentence_reset_time:
                sentence = ""
                sentence_reset_time = 0    
                
            
            # Display a message
            cv2.putText(image, "No hands detected", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Always display the current sentence
        cv2.putText(image, f'Sentence:{sentence}', (20, 450), 
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