import tensorflow_datasets as tfds

import os
print(os.path.abspath("droid_dataset/droid_100/1.0.0/features.json"))
print(os.path.exists("droid_dataset/droid_100/1.0.0/features.json"))


builder = tfds.builder_from_directory(builder_dir="droid_dataset/droid_100/1.0.0/")

print("Dataset Features:")
print(builder.info.features)