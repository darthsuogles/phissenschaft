""" Compute receptive field size

    http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf
"""

def receptive_field_size(layers):
    """ Find the receptive field size of a single 'neuron'
        after a series of 'conv' and 'pool' operations
    """
    if not layers: return 0
    for layer in layers:
        _ns = len(layer); assert _ns >= 2, layer; _tp = layer[0];
        if 'conv' == _tp: assert _ns >= 3, layer
        elif 'pool' == _tp: pass
        else: raise ValueError('unrecognized layer type: {}'.format(_tp))

    f_p = 1; s_p = 1;
    for layer in layers:
        layer_type, f = layer[:2]
        if 'conv' == layer_type:
            s = layer[2]
        elif 'pool' == layer_type:
            s = f            

        """ 
        Illustrating one kernel applied to previous image,
        to generate a single pixel in the resulting image.
        The dots represents images and 
        the strides represents distance        
        corresponding to the original image pixels
        |<-  f-1  ->|
        o--o--o--o--o--o--
        |  |  |  |  |  |
        """
        f_p += (f - 1) * s_p
        s_p *= s            
                    
    return f_p, s_p

# AlexNet sanf FC layers
r, s = receptive_field_size([
    ('conv', 11, 4), 
    ('pool', 2),
    ('conv', 5, 1), 
    ('pool', 2),
    ('conv', 3, 1), 
    ('conv', 3, 1),
    ('conv', 3, 1),
    ('pool', 2)
])
print('receptive_field_size', r, 'stride', s)

'''
ooo
  ooo
    ooo
      ooo
'''

print(receptive_field_size([('conv', 3, 2), ('conv', 3, 2)]))
