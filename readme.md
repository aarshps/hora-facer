# Face Recognition Project

This project uses a K-Nearest Neighbors model to predict whether the user likes or dislikes a given image based on their past preferences.

## Requirements

- Python 3.7 or higher
- OpenCV
- scikit-learn
- matplotlib

You can install the required packages using pip:

```bash
pip install opencv-python scikit-learn matplotlib
```

## Usage

1. Place your training images in the `data/train` directory. The images should be organized into two subdirectories: `like` and `dislike`.

2. Place the images you want to predict in the `data/predict` directory.

3. Run the main script:

```bash
python main.py
```

The script will train a K-Nearest Neighbors model on the training images, then use the model to predict whether you would like or dislike the images in the `data/predict` directory. After each prediction, you will be asked to confirm whether the prediction was correct.

The model will be saved after each run, so it can learn from your preferences over time. If you want to reset the model, you can do so when prompted at the start of the script.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)