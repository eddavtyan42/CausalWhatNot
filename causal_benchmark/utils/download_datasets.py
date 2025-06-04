from utils.loaders import load_dataset


def main():
    for name in ['asia', 'sachs', 'alarm', 'child']:
        load_dataset(name, n_samples=10000, force=True)


if __name__ == '__main__':
    main()
