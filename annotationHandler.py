import json
import PIL
import numpy as np
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
bb_params = ['height', 'width', 'x', 'y']


def convert_bb_size(bb, desired_size, from_size):
    bb = [bb[p] for p in bb_params]
    conv_x = (float(desired_size[0]) / float(from_size[0]))
    conv_y = (float(desired_size[1]) / float(from_size[1]))
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb

def create_bbx(path, desired_size, filenames, filename_path):
    bb_json = {}
    for c in anno_classes:
        j = json.load(open('{}/{}_labels.json'.format(path, c), 'r'))
        for l in j:
            if 'annotations' in l.keys() and len(l['annotations'])>0:
                bb_json[l['filename'].split('/')[-1]] = sorted(
                    l['annotations'], key=lambda x: x['height']*x['width'])[-1]
    print('Annotations for {} images found.\nCreating blank bounding boxes for the rest.'.format(len(bb_json)))

    for f in filenames:
        if not f.split("/")[-1] in bb_json.keys(): bb_json[f.split("/")[-1]] = empty_bbox

    sizes = [PIL.Image.open('{}/{}/{}'.format(filename_path, f.split("/")[-2],f.split("/")[-1])).size for f in filenames]

    return np.stack([convert_bb_size(bb_json[f.split("/")[-1]], desired_size, s) for f,s in zip(filenames, sizes)],).astype(np.float32)
