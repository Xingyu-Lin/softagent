
def get_preprocessor(type):
    if type is None:
        return lambda x: x # Identity
    elif type == 'image':
        def image_preprocess(x):
            x = x.permute(0, 3, 1, 2).contiguous()
            x /= 255 # to [0, 1]
            x = 2 * x - 1 # to [-1, 1]
            return x
        return image_preprocess
    else:
        raise ValueError(type)

