# TrashNet dataset setup

Download dataset-resized.zip from the [TrashNet authors' dataset page](https://huggingface.co/datasets/garythung/trashnet/tree/main), linked by the [original repository](https://github.com/garythung/trashnet). Extract it here so there is no extra nested dataset-resized folder:

~~~text
data/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
~~~

The original archive has 2,527 images. Credit Gary Thung and Mindy Yang and retain the upstream [MIT license notice](https://github.com/garythung/trashnet/blob/master/LICENSE) when redistributing dataset material. The full dataset is not included in this repository.

Run python prepare_data.py from the project root using the installed project environment. The code excludes six conflicting duplicate files without deleting them and reuses results/splits.json: 1,764 training, 378 validation, and 379 test images. Do not rename, replace, or add images when reproducing the recorded results. Only this guide is included in Git; downloaded images and archives remain local.
