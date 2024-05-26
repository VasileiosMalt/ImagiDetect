import os
import numpy as np
import joblib
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import shutil
import subprocess

def load_data(data_source):
    img_rows, img_cols = 224, 224
    batch_size = 1
    image_generator = ImageDataGenerator(rescale=1/255.0)
    test_generator = image_generator.flow_from_directory(
        directory=data_source,
        class_mode=None,
        batch_size=batch_size,
        target_size=(img_rows, img_cols),
        color_mode="rgb",
        shuffle=False,
        classes=['.']  # This ensures it looks for images in the main directory without subfolders
    )
    print(f"Loaded {test_generator.n} images from {data_source}")
    return test_generator

def load_feature_extractor():
    extractormodel = ResNet50V2(weights='imagenet', include_top=False, pooling='avg')
    return extractormodel

def extract_features(data_generator, model):
    num_images = data_generator.n
    if num_images == 0:
        print("No images found in the test data directory.")
        return np.zeros((0, 2048))
    
    image_data = np.zeros([num_images, 2048])
    for i in range(num_images):
        try:
            tmp_image = next(data_generator)
            tmp_feature_vector = model.predict(tmp_image, verbose=0)
            image_data[i, :] = tmp_feature_vector
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    print(f"Extracted features for {num_images} images")
    return image_data

def main():
    parser = argparse.ArgumentParser(description='Inference script for image classification.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data folder')
    parser.add_argument('--prob_limit', type=float, required=True, help='Probability limit for classification')
    parser.add_argument('--no_supplementary', action='store_false', help='Do not run supplementary program after inference', dest='run_supplementary')
    args = parser.parse_args()

    test_data_source = args.test_data
    prob_limit = args.prob_limit
    root_folder = os.getcwd()

    if not os.path.exists(test_data_source):
        print(f"Test data source directory {test_data_source} does not exist.")
        return

    if not os.path.exists(root_folder):
        print(f"Root folder {root_folder} does not exist.")
        return

    # Load saved classifier and class labels
    classifier = joblib.load(os.path.join(root_folder, 'classifier.pkl'))
    class_labels = np.load(os.path.join(root_folder, 'class_labels.npy'), allow_pickle=True).item()
    
    # Ensure class_labels is a dictionary
    if not isinstance(class_labels, dict):
        print("class_labels.npy is not a dictionary. Exiting.")
        return

    # Create a list of class names based on the dictionary
    class_names = [None] * len(class_labels)
    for class_name, index in class_labels.items():
        class_names[index] = class_name

    
    
    # Load test data
    test_generator = load_data(test_data_source)
    
    # Load feature extractor
    feature_extractor = load_feature_extractor()
    
    # Extract features
    test_features = extract_features(test_generator, feature_extractor)
    
    if test_features.shape[0] == 0:
        print("No features extracted from the test data. Exiting.")
        return
    
    # Classify images
    predictions = classifier.predict_proba(test_features)
    
    # Determine the name of the output folder
    test_data_folder_name = os.path.basename(os.path.normpath(test_data_source))
    output_base_dir = os.path.join(root_folder, 'predictions', test_data_folder_name)
    
    # Organize results
    for i, prediction in enumerate(predictions):
        class_index = np.argmax(prediction)
        class_label = class_names[class_index]
        high_prob = prediction[class_index]
        low_prob = 1 - high_prob
        
        src_image_path = test_generator.filepaths[i]
        if high_prob > prob_limit:
            dest_dir = os.path.join(output_base_dir, 'HighProbability', class_label)
        else:
            dest_dir = os.path.join(output_base_dir, 'LowProbability', class_label)
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        dest_image_path = os.path.join(dest_dir, os.path.basename(src_image_path))
        shutil.copyfile(src_image_path, dest_image_path)
    
    print("Inference completed and images categorized successfully.")
    
    # Run supplementary program if the user has not disabled it
    if args.run_supplementary:
        subprocess.run(['python', 'supplementary_program.py', '--dest_name', test_data_folder_name])

if __name__ == "__main__":
    main()
