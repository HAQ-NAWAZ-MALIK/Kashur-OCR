# Kashur-OCR




# keep the downloaded files in root





```


import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt

class KashmiriTextScanner:
    def __init__(self, model_path='model_output/best_model.keras', mapping_path='model_output/char_to_idx.json'):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        
        print("Loading character mapping...")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
            self.idx_to_char = {int(v): k for k, v in self.char_to_idx.items()}

    def preprocess_line_image(self, image):
        """Preprocess the complete text line image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        return image

    def segment_characters(self, image):
        """Segment the text line into individual characters"""
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours from right to left (for Arabic/Kashmiri script)
        char_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 20:  # Filter out very small contours
                char_regions.append((x, y, w, h))
        
        # Sort right to left
        char_regions.sort(key=lambda x: x[0], reverse=True)
        
        return char_regions

    def recognize_character(self, char_image):
        """Recognize a single character"""
        # Resize to model input size
        char_image = cv2.resize(char_image, (64, 64))
        
        # Normalize
        char_image = char_image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        char_image = np.expand_dims(char_image, axis=-1)
        char_image = np.expand_dims(char_image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(char_image, verbose=0)[0]
        char_idx = np.argmax(predictions)
        confidence = float(predictions[char_idx])
        
        return self.idx_to_char[char_idx], confidence

    def scan_text_line(self, image):
        """Scan a complete line of text"""
        # Preprocess the image
        processed_image = self.preprocess_line_image(image)
        
        # Segment into characters
        char_regions = self.segment_characters(processed_image)
        
        # Recognize each character
        recognized_text = []
        confidences = []
        
        for x, y, w, h in char_regions:
            # Extract character image
            char_image = processed_image[y:y+h, x:x+w]
            
            # Recognize character
            char, conf = self.recognize_character(char_image)
            recognized_text.append(char)
            confidences.append(conf)
        
        return {
            'text': ''.join(recognized_text),
            'confidences': confidences,
            'char_regions': char_regions,
            'processed_image': processed_image
        }

    def display_results(self, original_image, results):
        """Display the recognition results"""
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(3, 1, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Processed image with character regions
        plt.subplot(3, 1, 2)
        processed_with_regions = cv2.cvtColor(results['processed_image'], cv2.COLOR_GRAY2RGB)
        for x, y, w, h in results['char_regions']:
            cv2.rectangle(processed_with_regions, (x, y), (x+w, y+h), (0, 255, 0), 1)
        plt.imshow(processed_with_regions)
        plt.title('Character Segmentation')
        plt.axis('off')
        
        # Text results
        plt.subplot(3, 1, 3)
        plt.text(0.1, 0.5, 
                f"Recognized Text: {results['text']}\n\n" + 
                f"Average Confidence: {np.mean(results['confidences']):.2%}\n" +
                f"Number of Characters: {len(results['text'])}",
                fontsize=12, verticalalignment='center')
        plt.axis('off')
        plt.title('Recognition Results')
        
        plt.tight_layout()
        plt.show()

def process_kashmiri_text(image_path):
    """Process a complete text line image"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        # Create scanner
        scanner = KashmiriTextScanner()
        
        # Scan text
        results = scanner.scan_text_line(image)
        
        # Display results
        scanner.display_results(image, results)
        
        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Interactive upload and scanning
from google.colab import files
import io
import os

def scan_uploaded_text():
    try:
        print("Please upload a Kashmiri text image...")
        uploaded = files.upload()
        
        for filename in uploaded.keys():
            print(f"\nProcessing image: {filename}")
            
            # Save uploaded file temporarily
            temp_path = f"temp_{filename}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded[filename])
            
            # Process image
            results = process_kashmiri_text(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            if results:
                print("\nRecognition Summary:")
                print("=" * 50)
                print(f"Recognized Text: {results['text']}")
                print(f"Average Confidence: {np.mean(results['confidences']):.2%}")
                print(f"Number of Characters: {len(results['text'])}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the interactive scanner
if __name__ == "__main__":
    print("Improved Kashmiri Text Scanner")
    print("=" * 50)
    scan_uploaded_text()
```
