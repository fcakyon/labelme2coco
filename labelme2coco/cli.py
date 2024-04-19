import fire

from labelme2coco import convert


def app() -> None:
    """Cli app."""
    fire.Fire(lambda labelme_folder, export_dir="runs/labelme2coco/", train_split_rate=1, skip_labels=[], category_id_start=0: convert(labelme_folder, export_dir, train_split_rate, skip_labels, category_id_start))


if __name__ == "__main__":
    app()
