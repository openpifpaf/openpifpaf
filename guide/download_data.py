import openpifpaf


def main():
    openpifpaf.plugin.register()
    openpifpaf.plugins.cifar10.datamodule.Cifar10().download_data()
    openpifpaf.network.factory(checkpoint='shufflenetv2k16', download_progress=False)


if __name__ == '__main__':
    main()
