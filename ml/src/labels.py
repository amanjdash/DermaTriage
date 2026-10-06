"""The canonical HAM10000 class order used by training and serving."""

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}


def encode_label(label: str) -> int:
    try:
        return CLASS_TO_IDX[label]
    except KeyError as exc:
        raise ValueError(f"Unknown HAM10000 label: {label!r}") from exc

