import os
import subprocess
import pytest


TRAIN_COMMAND = [
    'python3', '-m', 'openpifpaf.train',
    '--lr=1e-3',
    '--momentum=0.95',
    '--epochs=1',
    '--batch-size=1',
    '--basenet=resnet18', '--no-pretrain',
    '--head-quad=1',
    '--headnets', 'pif', 'paf', 'paf25',
    '--square-edge=161',
    '--lambdas', '10', '1', '1', '15', '1', '1', '15', '1', '1',
    '--train-annotations', 'tests/coco/train1.json',
    '--train-image-dir', 'tests/coco/images/',
    '--val-annotations', 'tests/coco/train1.json',
    '--val-image-dir', 'tests/coco/images/',
]


PREDICT_COMMAND = [
    'python3', '-m', 'openpifpaf.predict',
    '--output-types=json',
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
        '--output-directory={}'.format(tmp_path),
    ]
    print(' '.join(predict_cmd))
    subprocess.run(predict_cmd, check=True, capture_output=True)
    print(os.listdir(tmp_path))

    assert 'puppy_dog.jpg.pifpaf.json' in os.listdir(tmp_path)
