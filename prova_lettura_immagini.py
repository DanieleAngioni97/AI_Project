"""def read_pgm(pgmf):
    #Return a raster of integers from a PGM as a list of lists.
    assert pgmf.readline() == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster"""


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
    from matplotlib import pyplot
    image = read_pgm("DaimlerBenchmark/1/non-ped_examples/img_00000.pgm", byteorder='<')
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()
    

    """new_shape=(100,100)
    new_image = numpy.resize(image, new_shape)
    pyplot.imshow(new_image, pyplot.cm.gray)
    pyplot.show()"""

    """#basewidth = 30

    img = Image.open("DaimlerBenchmark/Data/TrainingData/NonPedestrians/neg00000.pgm")
    #wpercent = (basewidth / float(img.size[0]))
    #hsize = int((float(img.size[1]) * float(wpercent)))
    #img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    width = 30
    height = 30
    img = img.resize((width, height), Image.ANTIALIAS)
    #img.save('resized_image.jpg')
    pyplot.imshow(img, pyplot.cm.gray)
    pyplot.show()


    from numpy import asarray

    # convert image to numpy array
    data = asarray(img)
    print(type(data))
    # summarize shape
    print(data.shape)
    pyplot.imshow(data, pyplot.cm.gray)
    pyplot.show()"""

    new_shape = (30, 30)
    reshape_image = read_pgm_reshape("DaimlerBenchmark/1/non-ped_examples/img_00000.pgm", new_shape)
    pyplot.imshow(reshape_image, pyplot.cm.gray)
    pyplot.show()


