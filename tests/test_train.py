import os
import subprocess
import pytest


TRAIN_COMMAND = [
    'python3', '-m', 'openpifpaf.train',
    '--lr=1e-3',
    '--momentum=0.9',
    '--epochs=1',
    '--batch-size=1',
    '--basenet=resnet18', '--no-pretrain',
    '--head-quad=1',
    '--headnets', 'cif', 'caf', 'caf25',
    '--square-edge=161',
    '--cocokp-train-annotations', 'tests/coco/train1.json',
    '--coco-train-image-dir', 'tests/coco/images/',
    '--cocokp-val-annotations', 'tests/coco/train1.json',
    '--coco-val-image-dir', 'tests/coco/images/',
]


PREDICT_COMMAND = [
    'python3', '-m', 'openpifpaf.predict',
    'tests/coco/images/puppy_dog.jpg',
]


@pytest.mark.skipif(os.getenv('PIFPAFTRAINING') != '1', reason='env PIFPAFTRAINING is not set')
def test_train(tmp_path):
    # train a model
    train_cmd = TRAIN_COMMAND + ['--out={}'.format(os.path.join(tmp_path, 'train_test.pkl'))]
    print(' '.join(train_cmd))
    subprocess.run(train_cmd, check=True, capture_output=True)
    print(os.listdir(tmp_path))

    # find the trained model checkpoint
    final_model = next(iter(f for f in os.listdir(tmp_path) if f.endswith('.pkl')))

    # run a prediction with that model
    predict_cmd = PREDICT_COMMAND + [
        '--checkpoint={}'.format(os.path.join(tmp_path, final_model)),
        '--json-output={}'.format(tmp_path),
    ]
    print(' '.join(predict_cmd))
    subprocess.run(predict_cmd, check=True, capture_output=True)
    print(os.listdir(tmp_path))

    assert 'puppy_dog.jpg.predictions.json' in os.listdir(tmp_path)
