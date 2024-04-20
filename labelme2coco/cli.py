import fire

from labelme2coco import convert


def app() -> None:
    """Cli app."""
    fire.Fire(convert)


if __name__ == "__main__":
    app()
