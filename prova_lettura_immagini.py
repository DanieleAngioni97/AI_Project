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


if __name__ == "__main__":
    from matplotlib import pyplot
    image = read_pgm("DaimlerBenchmark/Data/TrainingData/NonPedestrians/neg00000.pgm", byteorder='<')
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()


    """new_shape=(100,100)
    new_image = numpy.resize(image, new_shape)
    pyplot.imshow(new_image, pyplot.cm.gray)
    pyplot.show()"""

    from PIL import Image

    basewidth = 30
    img = Image.open("DaimlerBenchmark/Data/TrainingData/NonPedestrians/neg00000.pgm")
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    #img.save('resized_image.jpg')
    pyplot.imshow(img, pyplot.cm.gray)
    pyplot.show()