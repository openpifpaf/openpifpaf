import openpifpaf


def main():
    openpifpaf.contrib.cifar10.datamodule.Cifar10().download_data()


if __name__ == '__main__':
    main()
