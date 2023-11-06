from conan_dataset import build_mc_dataset

dataset_path = "dataset/survival"

if __name__ == "__main__":

    mc_dataset = build_mc_dataset(dataset_path, "val")
    print(mc_dataset[0])