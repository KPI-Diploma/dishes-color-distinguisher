import json
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def main():
    generate_training_data()


def run_model(colors):
    model = load_model('my_model.h5')

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    colors = [hex_to_rgb(color) for color in colors]
    colors = np.array(colors).reshape(1, 3, 3)

    prediction = model.predict(colors)

    print(prediction)

    top_10 = np.argsort(prediction[0])[-10:]

    for i in reversed(top_10):
        print(le.classes_[i])


def create_model():
    df = pd.read_csv('color_dataset.csv')

    X = []

    for _, row in df.iterrows():
        rgb_colors = []
        for i in range(1, 6):
            rgb_colors.append(hex_to_rgb(row[f'color{i}']))
        X.append(rgb_colors)

    X = np.array(X)

    y = df['class']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, y_train = X, y

    model = Sequential()
    model.add(Flatten(input_shape=(3, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(le.classes_), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=16)

    model.save('my_model.h5')

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
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i + 2], 16) for i in range(0, len(hex_color), 2)]


if __name__ == '__main__':
    main()
