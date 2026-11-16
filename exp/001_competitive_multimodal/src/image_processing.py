from PIL import Image, ImageEnhance, ImageOps


def load_image(path, max_side, contrast=1.0):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        return image.copy()
