from data import pascal

if __name__ == "__main__":
    config_name = "MaskRCNN"
    voc_dir = "/media/imanoid/DATA/workspace/data/VOCdevkit/VOC2012"

    data_loader = pascal.PascalVocDataLoader(config_name, voc_dir)
    data_loader.initialize_data()
