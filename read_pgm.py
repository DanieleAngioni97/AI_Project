import re
import numpy
from PIL import Image


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def read_pgm_reshape(filename, new_shape):
    width = new_shape[0]
    height = new_shape[1]

    image = Image.open(filename)
    resize_image = image.resize((width, height), Image.ANTIALIAS)
    return resize_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # new_shape = (30, 30)
    # reshaped_image = read_pgm_reshape("data/INRIAPerson/70X134H96/Test/pos/crop001501a.png", new_shape)
    # pyplot.imshow(reshaped_image, pyplot.cm.gray)
    # pyplot.show()

    image = Image.open("data/INRIA_organized/Test/neg/00001147.png")
    plt.imshow(image)
    plt.show()

    print("")


