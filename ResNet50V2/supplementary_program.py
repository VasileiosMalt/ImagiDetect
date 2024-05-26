import os
import time
import shutil
import torch
import pathlib
import pandas as pd
from PIL import Image
import argparse
import logging
from ultralytics import YOLO

# Set logger level to WARNING to suppress info and debug messages
logging.getLogger('ultralytics').setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description='Supplementary program for post-processing classified images.')
    parser.add_argument('--dest_name', type=str, required=True, help='Name of the test data folder in the predictions directory')
    args = parser.parse_args()

    dest_name = args.dest_name

    print("/nCategory sorting initiating...")
    low_prob_fold = os.path.join(os.getcwd(), 'predictions', dest_name, "LowProbability")
    high_prob_fold = os.path.join(os.getcwd(), 'predictions', dest_name, "HighProbability")

    for folder_structure in [low_prob_fold, high_prob_fold]:
        source_dir = folder_structure
        subfolders = [f.path for f in os.scandir(folder_structure) if f.is_dir()]
        main_categories = ['1ProtestSelfies', '2Crowds', '3Groupies', '4Performances', '5ProtestMaterials', '6ArtificialProtestImages', '7Threat', '8Meeting_Deliberation', '9NeutralImages', '_Ambiguous']
        main_categories = [os.path.join(folder_structure, main) for main in main_categories]
        sub_categories = [x for x in subfolders if x not in main_categories]
        
        for subcat in sub_categories:
            if os.path.basename(subcat).startswith("1"):
                dest_path = os.path.join(folder_structure, '1ProtestSelfies')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("2"):
                dest_path = os.path.join(folder_structure, '2Crowds')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("3"):
                dest_path = os.path.join(folder_structure, '3Groupies')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("4"):
                dest_path = os.path.join(folder_structure, '4Performances')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("5"):
                dest_path = os.path.join(folder_structure, '5ProtestMaterials')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("6"):
                dest_path = os.path.join(folder_structure, '6ArtificialProtestImages')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("7"):
                dest_path = os.path.join(folder_structure, '7Threat')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("8"):
                dest_path = os.path.join(folder_structure, '8Meeting_Deliberation')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("9"):
                dest_path = os.path.join(folder_structure, '9NeutralImages')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            elif os.path.basename(subcat).startswith("_"):
                dest_path = os.path.join(folder_structure, '_Ambiguous')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.move(subcat, dest_path)
            else:
                pass
    print("/nCategory sorting ended")

    print("---Supplementary method is getting applied---")
    start_time = time.time()

    model = YOLO('yolov8m.pt', verbose=False)

    basic_folder_name = dest_name
    Probs = ['HighProbability', 'LowProbability']
    
    def files(folder, extensions=['JPG', 'jpg', 'png', 'jpeg']):
        walk_list = []
        for i in os.walk(folder):
            walk_list.append(os.path.abspath(i[0]))
        extensions = map(lambda s: '*.' + s, extensions)
        walk_list = [x for x in walk_list if "19ProtestFar" not in x]
        walk_list = [x for x in walk_list if "17AnimalProtest" not in x]
        all_files = []
        for subfolders in walk_list:
            for ext in extensions:
                all_files += pathlib.Path(subfolders).glob(ext)
        all_files = list(map(str, all_files))
        return all_files

    final_df = pd.DataFrame()

    for probability in Probs:
        folder = os.path.join(os.getcwd(), 'predictions', basic_folder_name, probability, '1ProtestSelfies')
        images = files(folder)

        try:
            os.mkdir(os.path.join(folder, "SuspectedFalsePositives"))
        except:
            pass

        for img in images:
            try:
                results = model(img)
                df = results.pandas().xyxy[0]
                df['image_full_path_when_detected'] = img
                df['image_name'] = os.path.basename(img)
                persons = []
                for i in df['name'].str.contains('person'):
                    if i:
                        persons.append(i)
                if len(persons) > 6:
                    tail = os.path.split(img)[1]
                    head = os.path.split(img)[0]
                    dest = os.path.join(head, 'SuspectedFalsePositives', tail)
                    shutil.move(img, dest)
                final_df = final_df.append(df, ignore_index=True)
                img = Image.open(img)
                w, h = img.size
                dim = w * h
                df['percent_of_image_covered_by_object'] = round(((df.ymax - df.ymin) * (df.xmax - df.xmin) / dim), 2)
                df = df[df['name'].str.contains('person')]
                df = df[df['percent_of_image_covered_by_object'] > 0.25]
                if img not in df['image_full_path_when_detected']:
                    tail = os.path.split(img)[1]
                    head = os.path.split(img)[0]
                    dest = os.path.join(head, 'SuspectedFalsePositives', tail)
                    shutil.move(img, dest)
            except:
                continue

        folder = os.path.join(os.getcwd(), 'predictions', basic_folder_name, probability, '2Crowds')
        images = files(folder)

        try:
            os.mkdir(os.path.join(folder, "SuspectedFalsePositives"))
        except:
            pass
        try:
            os.mkdir(os.path.join(folder, "SuspectedGroupies"))
        except:
            pass
        for img in images:
            try:
                results = model(img)
                df = results.pandas().xyxy[0]
                df['image_full_path_when_detected'] = img
                df['image_name'] = os.path.basename(img)
                persons = []
                for i in df['name'].str.contains('person'):
                    if i:
                        persons.append(i)
                if len(persons) < 2:
                    tail = os.path.split(img)[1]
                    head = os.path.split(img)[0]
                    dest = os.path.join(head, 'SuspectedFalsePositives', tail)
                    shutil.move(img, dest)
                elif 3 < len(persons) < 8:
                    tail = os.path.split(img)[1]
                    head = os.path.split(img)[0]
                    dest = os.path.join(head, 'SuspectedGroupies', tail)
                    shutil.move(img, dest)
                final_df = final_df.append(df, ignore_index=True)
            except:
                continue

        folder = os.path.join(os.getcwd(), 'predictions', basic_folder_name, probability, '3Groupies')
        images = files(folder)

        try:
            os.mkdir(os.path.join(folder, "SuspectedFalsePositives"))
        except:
            pass
        try:
            os.mkdir(os.path.join(folder, "SuspectedMeeting_Deliberation"))
        except:
            pass
        for img in images:
            try:
                results = model(img)
                df = results.pandas().xyxy[0]
                df['image_full_path_when_detected'] = img
                df['image_name'] = os.path.basename(img)
                persons = []
                for i in df['name'].str.contains('person'):
                    if i:
                        persons.append(i)
                if len(persons) < 2:
                    tail = os.path.split(img)[1]
                    head = os.path.split(img)[0]
                    dest = os.path.join(head, 'SuspectedFalsePositives', tail)
                    shutil.move(img, dest)
                chairs = []
                for i in df['name'].str.contains('chair'):
                    if i:
                        chairs.append(i)
                if len(chairs) > 2:
                    tail = os.path.split(img)[1]
                    head = os.path.split(img)[0]
                    dest = os.path.join(head, 'SuspectedMeeting_Deliberation', tail)
                    try:
                        shutil.move(img, dest)
                    except:
                        pass
                final_df = final_df.append(df, ignore_index=True)
            except:
                continue

        folder = os.path.join(os.getcwd(), 'predictions', basic_folder_name, probability, '8Meeting_Deliberation')
        images = files(folder)

        try:
            os.mkdir(os.path.join(folder, "SuspectedFalsePositives"))
        except:
            pass
        for img in images:
            try:
                results = model(img)
                df = results.pandas().xyxy[0]
                df['image_full_path_when_detected'] = img
                df['image_name'] = os.path.basename(img)
                persons = []
                for i in df['name'].str.contains('person'):
                    if i:
                        persons.append(i)
                if len(persons) < 4:
                    tail = os.path.split(img)[1]
                    head = os.path.split(img)[0]
                    dest = os.path.join(head, 'SuspectedFalsePositives', tail)
                    shutil.move(img, dest)
                final_df = final_df.append(df, ignore_index=True)
            except:
                continue

    stop_time = time.time()
    tot_time = (stop_time - start_time) / 60
    tot_time = round(tot_time, 1)
    print("---Supplementary method program was running %s minutes ---" % tot_time)
    print("All procedures terminated!")
    print("Results are located here: ")
    print(os.path.join(os.getcwd(), 'predictions', basic_folder_name))

if __name__ == "__main__":
    main()
