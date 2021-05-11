import os
import argparse


def create_dirs(path):
    history = os.path.join(path, 'history')
    models = os.path.join(path, 'models')
    results = os.path.join(path, 'results')
    os.mkdir(path)
    os.mkdir(history)
    os.mkdir(models)
    os.mkdir(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, help='Путь директории') 
    args = parser.parse_args()
    create_dirs(args.path)
