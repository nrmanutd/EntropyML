from pathlib import Path
from typing import Dict, Tuple
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive


_TINY_IMAGENET_URL = (
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
)
_TINY_IMAGENET_ARCHIVE = "tiny-imagenet-200.zip"
_TINY_IMAGENET_DIR = "tiny-imagenet-200"


class _TinyImageNetValDataset(Dataset):
    """
    Validation-split Tiny ImageNet.

    В отличие от train, изображения validation лежат в одной папке:
        val/images/

    Их классы записаны в:
        val/val_annotations.txt
    """

    def __init__(
        self,
        dataset_dir: Path,
        class_to_idx: Dict[str, int],
        transform=None,
    ):
        self.transform = transform

        val_dir = dataset_dir / "val"
        images_dir = val_dir / "images"
        annotations_file = val_dir / "val_annotations.txt"

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Не найдена директория validation-изображений: "
                f"{images_dir}"
            )

        if not annotations_file.is_file():
            raise FileNotFoundError(
                f"Не найден файл validation-разметки: "
                f"{annotations_file}"
            )

        self.samples = []

        with annotations_file.open("r", encoding="utf-8") as file:
            for line in file:
                # Формат строки:
                # filename  wnid  x0  y0  x1  y1
                parts = line.rstrip("\n").split("\t")

                if len(parts) < 2:
                    continue

                image_name = parts[0]
                wnid = parts[1]

                if wnid not in class_to_idx:
                    raise ValueError(
                        f"Неизвестный класс {wnid!r} в файле "
                        f"{annotations_file}"
                    )

                image_path = images_dir / image_name
                target = class_to_idx[wnid]

                self.samples.append((image_path, target))

        if not self.samples:
            raise RuntimeError(
                f"В файле {annotations_file} не найдено "
                f"ни одного validation-объекта"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        # default_loader автоматически приводит изображение к RGB.
        image = default_loader(str(image_path))

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def _prepare_tiny_imagenet(
    root: str,
    download: bool,
) -> Path:
    """
    Проверяет наличие Tiny ImageNet и при необходимости
    распаковывает или скачивает архив.

    root может указывать:
    1. на родительскую папку, например ./data;
    2. непосредственно на ./data/tiny-imagenet-200.
    """
    root_path = Path(root).expanduser().resolve()

    if root_path.name == _TINY_IMAGENET_DIR:
        dataset_dir = root_path
        download_root = root_path.parent
    else:
        dataset_dir = root_path / _TINY_IMAGENET_DIR
        download_root = root_path

    # Датасет уже распакован.
    if dataset_dir.is_dir():
        return dataset_dir

    download_root.mkdir(parents=True, exist_ok=True)

    archive_path = download_root / _TINY_IMAGENET_ARCHIVE

    # Позволяет вручную перенести архив на другой компьютер.
    if archive_path.is_file():
        print(f"Распаковка {archive_path}...")

        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(download_root)

    elif download:
        print(
            f"Скачивание Tiny ImageNet в директорию "
            f"{download_root}..."
        )

        try:
            download_and_extract_archive(
                url=_TINY_IMAGENET_URL,
                download_root=str(download_root),
                filename=_TINY_IMAGENET_ARCHIVE,
                remove_finished=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Не удалось автоматически скачать Tiny ImageNet. "
                f"Скачайте файл {_TINY_IMAGENET_ARCHIVE} вручную "
                f"и положите его в директорию {download_root}."
            ) from exc

    else:
        raise FileNotFoundError(
            f"Tiny ImageNet не найден в {dataset_dir}. "
            f"Положите архив {_TINY_IMAGENET_ARCHIVE} "
            f"в директорию {download_root} либо передайте "
            f"download=True."
        )

    if not dataset_dir.is_dir():
        raise RuntimeError(
            "Архив был обработан, но директория датасета "
            f"не появилась: {dataset_dir}"
        )

    return dataset_dir


def _dataset_to_tensors(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Последовательно переносит Dataset в два тензора.

    Используется предварительное выделение памяти, чтобы не хранить
    список из 100 000 отдельных float32-тензоров.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    X = None
    y = torch.empty(
        len(dataset),
        dtype=torch.int64,
    )

    offset = 0

    for xb, yb in loader:
        if X is None:
            X = torch.empty(
                (len(dataset), *xb.shape[1:]),
                dtype=xb.dtype,
            )

        end = offset + xb.shape[0]

        X[offset:end].copy_(xb)
        y[offset:end].copy_(yb.to(torch.int64))

        offset = end

    if X is None:
        raise RuntimeError("Dataset оказался пустым")

    if offset != len(dataset):
        raise RuntimeError(
            f"Загружено {offset} объектов, "
            f"ожидалось {len(dataset)}"
        )

    return X, y


def loadTinyImageNet_torch(
    root="./data",
    download=True,
    batch_size=512,
    num_workers=0,
):
    """
    Загружает Tiny ImageNet-200 полностью в оперативную память.

    Возвращает:
        Xtr: [100000, 3, 64, 64], float32 в [0,1]
        ytr: [100000], int64

        Xte: [10000, 3, 64, 64], float32 в [0,1]
        yte: [10000], int64

    Важно:
        Официальный test-split Tiny ImageNet не содержит открытых
        меток классов. Поэтому в качестве Xte/yte возвращается
        официальный validation-split.

        Нормализация к среднему и стандартному отклонению здесь
        не выполняется.
    """
    dataset_dir = _prepare_tiny_imagenet(
        root=root,
        download=download,
    )

    transform = transforms.ToTensor()

    # Структура train совместима с ImageFolder:
    #
    # train/
    #   n01443537/
    #       images/
    #   n01629819/
    #       images/
    #   ...
    ds_tr = datasets.ImageFolder(
        root=dataset_dir / "train",
        transform=transform,
    )

    # Validation лежит в одной папке, поэтому используем
    # собственный Dataset и ту же class_to_idx, что у train.
    ds_te = _TinyImageNetValDataset(
        dataset_dir=dataset_dir,
        class_to_idx=ds_tr.class_to_idx,
        transform=transform,
    )

    if len(ds_tr.classes) != 200:
        raise RuntimeError(
            f"Ожидалось 200 классов, "
            f"обнаружено {len(ds_tr.classes)}"
        )

    if len(ds_tr) != 100_000:
        raise RuntimeError(
            f"Ожидалось 100000 train-объектов, "
            f"обнаружено {len(ds_tr)}"
        )

    if len(ds_te) != 10_000:
        raise RuntimeError(
            f"Ожидалось 10000 validation-объектов, "
            f"обнаружено {len(ds_te)}"
        )

    print("Преобразование train-split в тензоры...")

    Xtr, ytr = _dataset_to_tensors(
        dataset=ds_tr,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("Преобразование validation-split в тензоры...")

    Xte, yte = _dataset_to_tensors(
        dataset=ds_te,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("Tiny ImageNet успешно загружен:")
    print(f"Xtr: {tuple(Xtr.shape)}, {Xtr.dtype}")
    print(f"ytr: {tuple(ytr.shape)}, {ytr.dtype}")
    print(f"Xte: {tuple(Xte.shape)}, {Xte.dtype}")
    print(f"yte: {tuple(yte.shape)}, {yte.dtype}")

    return Xtr, ytr, Xte, yte