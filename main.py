from sklearn.cluster import KMeans
import cv2
import json


def main():
    train_file = 'train.json'
    test_file = 'test.json'
    x = transform_data(train_file)
    print(x[:3])


def get_image_colors(image, n_colors):
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
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            colors = get_image_colors(image, 3)

            x.append(colors)
            y.append(food)

    return x, y


if __name__ == '__main__':
    main()
