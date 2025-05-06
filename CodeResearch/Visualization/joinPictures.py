from PIL import Image

def join_side_by_side(img_path_left: str,
                      img_path_right: str,
                      output_path: str = "combined.png") -> None:
    """
    Combines two PNG images into one, placing them side-by-side.

    Parameters
    ----------
    img_path_left : str
        Path to the left-hand image.
    img_path_right : str
        Path to the right-hand image.
    output_path : str, optional
        Where to save the combined PNG (default: 'combined.png').
    """
    # Load images
    img_left  = Image.open(img_path_left).convert("RGBA")
    img_right = Image.open(img_path_right).convert("RGBA")

    # Make sure they share the same size; otherwise resize the right one
    if img_left.size != img_right.size:
        img_right = img_right.resize(img_left.size, Image.LANCZOS)

    w, h = img_left.size
    new_canvas = Image.new("RGBA", (2 * w, h), (255, 255, 255, 0))

    # Paste images onto the new canvas
    new_canvas.paste(img_left,  (0,    0))
    new_canvas.paste(img_right, (w,    0))

    # Save result
    new_canvas.save(output_path, format="PNG")
    print(f"Combined image saved to '{output_path}'")

# Example usage:
#directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\PValueLogs_TargetTasks"
directory = "C:\\Users\\nrman\\YandexDisk\\Documents\\Научная работа\\Сложность\\Конференции\\NeurIPS 2025 09-12 Dec\\Paper\\MNIST_CIFAR Simple_Complex Pictures"
join_side_by_side(f"{directory}\\Slide1.png", f"{directory}\\Slide2.png", f"{directory}\\side_by_side_mnist_cifar.png")