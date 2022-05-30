"""Functions to get type and shape of various objects for testing
Adapted from: http://thinkpython2.com
"""

from __future__ import print_function, division


def getshape(ds):
    """Returns a string that describes the shape of a data structure.

    ds: any Python object

    Returns: string
    """
    typename = type(ds).__name__

    # handle sequences
    sequence = (list, tuple, set, type(iter('')))
    if isinstance(ds, sequence):
        # check if list empty:
        if not ds:
            return 'empty %s' % typename
        else:

            t = []
            for i, x in enumerate(ds):
                t.append(getshape(x))
            rep = '%s of %s' % (typename, listrep(t))
            return rep

    # handle dictionaries
    elif isinstance(ds, dict):
        keys = set()
        vals = set()
        for k, v in ds.items():
            keys.add(getshape(k))
            vals.add(getshape(v))
        rep = '%s of %d %s->%s' % (typename, len(ds),
                                   setrep(keys), setrep(vals))
        return rep

    # handle np arrays
    elif hasattr(ds, 'shape'):
        shape = ds.shape
        if len(shape) == 0:
            return 'empty %s' % typename
        elif len(shape) == 1:
            return '1D array of %d %s' % (shape[0], typename)
        elif len(shape) == 2:
            return '2D array of %d x %d %s' % (shape[0], shape[1], typename)
        else:
            return 'array of %s with shape=%s' % (typename, shape)

    # handle scalars
    elif ds is int:
        return typename

    # handle other types
    else:
        if hasattr(ds, '__class__'):
            return ds.__class__.__name__
        else:
            return typename


def listrep(t):
    """Returns a string representation of a list of type strings.

    t: list of strings

    Returns: string
    """
    current = t[0]
    count = 0
    res = []
    for x in t:
        if x == current:
            count += 1
        else:
            append(res, current, count)
            current = x
            count = 1
    append(res, current, count)
    return setrep(res)


def setrep(s):
    """Returns a string representation of a set of type strings.

    s: set of strings

    Returns: string
    """
    rep = ', '.join(s)
    if len(s) == 1:
        return rep
    else:
        return '(' + rep + ')'
    return


def append(res, typestr, count):
    """Adds a new element to a list of type strings.

    Modifies res.

    res: list of type strings
    typestr: the new type string
    count: how many of the new type there are

    Returns: None
    """
    if count == 1:
        rep = typestr
    else:
        rep = '%d %s' % (count, typestr)
    res.append(rep)


if __name__ == '__main__':

    t = [1, 2, 3]
    print(getshape(t))

    t2 = [[1, 2], [3, 4], [5, 6]]
    print(getshape(t2))

    t3 = [1, 2, 3, 4.0, '5', '6', [7], [8], 9]
    print(getshape(t3))

    class Point:
        """trivial object type"""

    t4 = [Point(), Point()]
    print(getshape(t4))

    s = set('abc')
    print(getshape(s))

    lt = zip(t, s)
    print(getshape(lt))

    d = dict(lt)
    print(getshape(d))

    it = iter('abc')
    print(getshape(it))
