import json
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def main():
    create_model()
    # run_model(['#ffffff', '#ffffff', '#ffffff'])


def run_model(colors):
    assert len(colors) == 3, "Input should be an array of 3 colors"

    model = load_model('model.h5')

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    colors_rgb = [hex_to_rgb(color) for color in colors]
    colors_flat = [component for color in colors_rgb for component in color]

    input_df = pd.DataFrame([colors_flat],
                            columns=['color1_r', 'color1_g', 'color1_b', 'color2_r', 'color2_g', 'color2_b', 'color3_r',
                                     'color3_g', 'color3_b'])

    predictions = model.predict(input_df.values)[0]
    top_10_indices = np.argsort(predictions)[::-1][:10]

    return le.inverse_transform(top_10_indices)


def create_model():
    df = pd.read_csv('color_dataset.csv')

    df['color1'] = df['color1'].apply(hex_to_rgb)
    df['color2'] = df['color2'].apply(hex_to_rgb)
    df['color3'] = df['color3'].apply(hex_to_rgb)

    colors_df = pd.DataFrame(df['color1'].tolist(), columns=['color1_r', 'color1_g', 'color1_b'])
    colors_df[['color2_r', 'color2_g', 'color2_b']] = pd.DataFrame(df['color2'].tolist())
    colors_df[['color3_r', 'color3_g', 'color3_b']] = pd.DataFrame(df['color3'].tolist())

    le = LabelEncoder()
    classes = le.fit_transform(df['class'])

    model = Sequential()
    model.add(Dense(16, input_dim=colors_df.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(le.classes_), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    model.fit(colors_df.values, classes, epochs=50, batch_size=32)

    model.save('model.h5')

    # Save the LabelEncoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)


def generate_training_data():
    with open('meta/classes.txt', 'r') as f:
        classes = [line.strip() for line in f]

    with open('meta/train.json', 'r') as f:
        train_data = json.load(f)

    x = []
    y = []

    for class_name in classes:
        print(f"Processing {class_name} images...")
        image_paths = train_data[class_name]
        for image_path in image_paths:
            full_image_path = os.path.join('images', f"{image_path}.jpg")
            colors = get_image_colors(full_image_path, 3)
            x.append(colors)
            y.append(class_name)
            df = pd.DataFrame(x, columns=['color1', 'color2', 'color3'])
            df['class'] = y
            df.to_csv('color_dataset.csv', index=False)


def get_image_colors(img_path, n_colors):
    image = cv2.imread(img_path)
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_

    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(c[2]), int(c[1]), int(c[0])) for c in colors]

    return hex_colors


def hex_to_rgb(hex_color):
    return [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]


if __name__ == '__main__':
    main()
