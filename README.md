# Hand-Gesture Tetris

Play Tetris with hand gestures from a normal webcam. A custom gesture classifier recognises six hand poses in real time and turns them into game controls.

Computer Vision course project at Technische Hochschule Deggendorf (03/2025 – 07/2025).

## How it works

1. **MediaPipe Hands** finds 21 hand landmarks in each webcam frame.
2. The landmarks are converted to coordinates relative to the wrist and **normalised**, so the features don't depend on where the hand is in the frame or how far it is from the camera.
3. A small **TensorFlow** classifier (exported to **TensorFlow Lite** for fast inference) maps the 42 features to one of six gestures.
4. The recognised gesture is sent to the **Pygame** Tetris loop, with a short cooldown so one gesture doesn't trigger several moves.

## Controls

| Gesture | Action |
|---|---|
| Index finger | Move left |
| Pinkies | Move right |
| Peace | Rotate |
| OK | Drop faster |
| Any gesture | Start the game |

The model also recognises **Open** and **Close** hand poses.

## The gesture model

| | |
|---|---|
| Gestures | Open, Close, Index Finger, OK, Peace, Pinkies |
| Training data | 5,940 landmark samples, collected with `app.py` |
| Split | 75% train / 25% test (random seed 42) |
| Model | MLP: 42 inputs → Dense 20 → Dense 10 → softmax, with dropout |
| Training | Early stopping (patience 20) |
| **Test accuracy** | **95%** on 1,485 held-out samples (macro F1 0.95) |

Training and evaluation, including the confusion matrix, are in [`keypoint_classification_EN.ipynb`](keypoint_classification_EN.ipynb).

## Run it

Requires Python 3.10–3.12 and a webcam.

```bash
pip install -r requirements.txt
python tetris.py
```

### Collect more training data

```bash
python app.py
```

Press `k` to switch to logging mode, then hold a gesture and press its number key (`0`–`5`) to save landmark samples to `model/keypoint_classifier/keypoint.csv`. Retrain with the notebook afterwards.

## Project structure

```
tetris.py                          Pygame Tetris game controlled by gestures
gesture_detector.py                Webcam + MediaPipe + classifier wrapper used by the game
app.py                             Live gesture viewer and training-data logger
keypoint_classification_EN.ipynb   Model training and evaluation
model/keypoint_classifier/         Training data, trained model (.keras, .hdf5, .tflite) and labels
utils/cvfpscalc.py                 FPS counter
```

## Team

- **Htet Aung Lin**: all gesture model development and training (data collection, preprocessing, training and evaluation)
- **Teammate**: game UI in Pygame

## Credits and license

The landmark pipeline, data logger and training notebook are adapted from [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) by Kazuhito Takahashi, licensed under Apache-2.0. This project uses its own gesture set, training data and trained model, and adds the Tetris game. It is distributed under the same [Apache-2.0 license](LICENSE).
