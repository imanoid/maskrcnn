from data import pascal

if __name__ == "__main__":
    config_name = "MaskRCNN"
    voc_dir = "/media/imanoid/Data/workspace/data/VOCdevkit/VOC2012"

    data_loader = pascal.PascalVocDataLoader(config_name, voc_dir)
    sample_names = data_loader.load_sample_names()
    for sample_name in sample_names:
        sample = data_loader._load_sample(sample_name)
        if "labels" in sample:
            print("breakpoint")

