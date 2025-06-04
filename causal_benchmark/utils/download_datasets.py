from utils.loaders import load_dataset


def main():
    load_dataset('asia', n_samples=10000, force=True)


if __name__ == '__main__':
    main()
