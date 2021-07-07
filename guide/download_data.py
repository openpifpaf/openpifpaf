import openpifpaf


def main():
    openpifpaf.plugins.cifar10.datamodule.Cifar10().download_data()
    openpifpaf.network.Factory(checkpoint='shufflenetv2k16', download_progress=False).factory()
    openpifpaf.network.Factory(checkpoint='shufflenetv2k30-wholebody',
                               download_progress=False).factory()
    openpifpaf.network.Factory(checkpoint='resnet50-crowdpose', download_progress=False).factory()
    openpifpaf.network.Factory(checkpoint='shufflenetv2k30-animalpose',
                               download_progress=False).factory()
    openpifpaf.network.Factory(checkpoint='shufflenetv2k16-apollo-24',
                               download_progress=False).factory()


if __name__ == '__main__':
    main()
