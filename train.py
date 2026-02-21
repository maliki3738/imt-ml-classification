import argparse
from pathlib import Path

import tensorflow as tf


def build_datasets(
    data_dir: Path,
    img_size=(224, 224),
    batch_size=16,
    validation_split=0.2,
    seed=42,
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preprocessing and split")
    parser.add_argument("--data-dir", type=Path, default=Path("../data/train"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    train_ds, val_ds = build_datasets(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    print("Classes détectées:", train_ds.class_names)
    print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())


if __name__ == "__main__":
    main()