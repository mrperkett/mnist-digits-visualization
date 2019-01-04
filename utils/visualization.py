"""
"""

import numpy as np


def add_image(img, canvas, xpos=0, ypos=0, color=None, partial_placements=True):
    """
    """
    img_xdim, img_ydim = img.shape
        
    if color is None:
        canvas_xdim, canvas_ydim = canvas.shape
    else:
        canvas_xdim, canvas_ydim, rgb = canvas.shape
        assert rgb == 3

    #import ipdb; ipdb.set_trace()
    if xpos > canvas_xdim or ypos > canvas_ydim:
        raise AssertionError("Placement location is out of bounds")

    if not partial_placements and (xpos + img_xdim > canvas_xdim or ypos + img_ydim > canvas_ydim):
        raise AssertionError("Placement of image will result in it being partially out of bounds.  Set partial_placement to True if this is acceptable behavior")


    for x in range(img_xdim):
        for y in range(img_ydim):

            # ignore the white background
            if img[x,y] == 0:
                continue

            # position of pixel on canvas
            x_canvas = xpos + x
            y_canvas = ypos + y

            # if the new point is in-bounds, add to the pixel value
            if x_canvas < canvas_xdim and y_canvas < canvas_ydim:
                # if grayscale
                if color is None:
                    canvas[x_canvas, y_canvas] = max(canvas[x_canvas, y_canvas], img[x, y])

                # if RGB
                else:
#                    if img[x, y] == 0:
#                        canvas[x_canvas, y_canvas] = np.array([255, 255, 255])
#                    else:
#                        canvas[x_canvas, y_canvas] = img[x, y] * color
#                    canvas[x_canvas, y_canvas] = (255 - img[x, y]) * color
                    canvas[x_canvas, y_canvas] = 255 * color + (255 - img[x, y]) * (1 - color)
                    

    return


def build_digit_image_data(images, data, labels, density=0.5, colors=None):
    """
    Args:
        images:
        data:
        labels:
        density:
        colors:
    """
    if len(images) != len(data) != len(labels):
        raise AssertionError("Length of images, data, and labels must be identical!")

    grayscale = True if colors is None else False
    avg_img_xdim = 28
    avg_img_ydim = 28

    # Initialize the canvas (data for full image)
    num_digits = len(images)
    canvas_xdim = int(np.ceil(np.sqrt(np.ceil(num_digits * avg_img_xdim * avg_img_ydim / density))))
    canvas_ydim = canvas_xdim
    
    if grayscale:
        canvas = np.full((canvas_ydim, canvas_xdim), 0, dtype='uint8')
    else:
        canvas = np.full((canvas_ydim, canvas_xdim, 3), 255, dtype='uint8')
    
    # scale t-SNE mapping between 0.0 and 1.0 (and swap x and y for the plot)
    mn = min(data.ravel())
    mx = max(data.ravel())
    norm_coordinates = (data - mn) / (mx - mn)
    norm_coordinates = norm_coordinates[:,[1,0]]
    
    # place digits on canvas
    for img, (norm_x, norm_y), label in zip(images, norm_coordinates, labels):
        img_ydim, img_xdim = img.shape
        xpos = int(norm_x * (canvas_xdim - avg_img_xdim))
        ypos = int(norm_y * (canvas_ydim - avg_img_ydim))
    
        if grayscale:
            rgb=None
        else:
            rgb = colors[label]
    
        add_image(img, canvas, xpos=xpos, ypos=ypos, color=rgb, partial_placements=False)

    return canvas
