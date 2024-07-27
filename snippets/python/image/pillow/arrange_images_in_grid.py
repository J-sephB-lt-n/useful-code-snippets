"""
TAGS: combine|grid|image|images|layout|pil|pillow|stack
DESCRIPTION: Combine multiple images (in PIL.Image format) into a single image by laying them out in a stack or grid 
REQUIREMENTS: pip install Pillow
"""

from PIL import Image


def arrange_images_in_grid(
    images: tuple[Image.Image, ...],
    layout: tuple[int, ...],
    background_colour: float | tuple[float, ...] | str | None = "black",
) -> Image.Image:
    """
    Combines multiple images into a single image by laying them out in a stack or grid

    Args:
        images (tuple): Tuple containing the PIL.Images to combine
        layout (tuple): Tuple of integers describing the layout
                        See examples under `Notes` below
        background_colour (float|tuple[float]|str|None): Background colour of merged image
                                                         This value is passed to argument `color` of PIL.Image.new()

    Notes:
        Example layouts:
            layout=(4,)
                +---+---+---+---+
                |   |   |   |   |
                +---+---+---+---+

            layout=(3,1,2)
                +---+---+---+
                |   |   |   |
                +---+---+---+
                |           |
                +-----+-----+
                |     |     |
                +-----+-----+

    Returns:
        PIL.Image.Image

    Examples:
        >>> from PIL import Image
        >>> example_images: tuple[Image.Image, ...] = (
        ...     Image.new(mode="RGB", size=(200,200), color="red"),
        ...     Image.new(mode="RGB", size=(140,235), color="blue"),
        ...     Image.new(mode="RGB", size=(300,150), color="orange"),
        ...     Image.new(mode="RGB", size=(300,300), color="green"),
        ...     Image.new(mode="RGB", size=(110,120), color="purple"),
        ...     Image.new(mode="RGB", size=(300,100), color="pink"),
        ... )
        >>> combined_image: Image.Image=arrange_images_in_grid(images=example_images, layout=(6,))
        >>> combined_image: Image.Image=arrange_images_in_grid(images=example_images, layout=(3,3))
        >>> combined_image: Image.Image=arrange_images_in_grid(images=example_images, layout=(2,2,2))
        >>> combined_image: Image.Image=arrange_images_in_grid(images=example_images, layout=(1,2,3))
        >>> combined_image.show()
    """
    if sum(layout) != len(images):
        raise ValueError(
            f"Layout {layout} does not match the number of images provided ({len(images)})"
        )

    # separate the list of images into a list of tuples of images (1 tuple per row) #
    image_rows: list[tuple[Image.Image, ...]] = []
    index: int = 0
    for row_len in layout:
        image_rows.append(images[index : index + row_len])
        index += row_len

    # Make each row of images into a single image #
    row_images: list[Image.Image] = []
    for image_row in image_rows:
        row_canvas = Image.new(
            mode="RGB",
            size=(
                sum(
                    [image.size[0] for image in image_row]
                ),  # row width = sum of widths of all images in row
                max(
                    [image.size[1] for image in image_row]
                ),  # row height = height of tallest image in row
            ),
            color=background_colour,
        )
        current_pixel_start = 0
        for image in image_row:
            row_canvas.paste(image, (current_pixel_start, 0))
            current_pixel_start += image.size[0]
        row_images.append(row_canvas)

    # stack row images vertically to create the final image #
    final_canvas = Image.new(
        mode="RGB",
        size=(
            max(
                [image.size[0] for image in row_images]
            ),  # row width = width of widest image in stack
            sum(
                [image.size[1] for image in row_images]
            ),  # row height = sum of heights of all stacked images
        ),
        color=background_colour,
    )
    current_pixel_start = 0
    for row_image in row_images:
        final_canvas.paste(row_image, (0, current_pixel_start))
        current_pixel_start += row_image.size[1]

    return final_canvas
