import os
from sklearn.cluster import KMeans
import cv2
import json
import pandas as pd


def main():
    generate_training_data()


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
            colors = get_image_colors(full_image_path, 5)
            x.append(colors)
            y.append(class_name)

    df = pd.DataFrame(x, columns=['color1', 'color2', 'color3', 'color4', 'color5'])
    df['class'] = y
    df.to_csv('color_dataset.csv', index=False)


def get_image_colors(img_path, n_colors):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_

    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(c[2]), int(c[1]), int(c[0])) for c in colors]

    return hex_colors


def transform_data(file):
    with open('meta/' + file, 'r') as f:
        train_data = json.load(f)

    x = []
    y = []

    for food, images in train_data.items():
        for img_path in images:
            img_path = f'images/{img_path}.jpg'
            colors = get_image_colors(img_path, 5)

            x.append(colors)
            y.append(food)

            print(food, colors)

    return x, y


if __name__ == '__main__':
    main()
