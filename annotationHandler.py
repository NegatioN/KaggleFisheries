import json
import PIL
import numpy as np
import copy
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
bb_params = ['height', 'width', 'x', 'y']


def save_as_sloth_format(bboxes, viewed_size, filenames, filename_path):
    sizes = [PIL.Image.open('{}/{}/{}'.format(filename_path, f.split("/")[-2],f.split("/")[-1])).size for f in filenames]
    output = []
    for i, (fname, s) in enumerate(zip(filenames, sizes)):
        entry = {}
        entry['class'] = 'image'
        entry['filename'] = fname.split("/")[-1]
        annotations = {}
        annotations['class'] = 'rect'
        print(bboxes[i])
        bb = convert_bb_size(bboxes[i], s, viewed_size)
        annotations['height'], annotations['width'], annotations['x'], annotations['y'] = bb
        entry['annotations'] = annotations
        output.append(entry)
    return output

def convert_bb_from_json(json_bb, desired_size, from_size):
    bb = [json_bb[p] for p in bb_params]
    return convert_bb_size(bb, desired_size, from_size)


def convert_bb_size(bb, desired_size, from_size):
    loc_bb = copy.copy(bb)
    conv_x = (float(desired_size[0]) / float(from_size[0]))
    conv_y = (float(desired_size[1]) / float(from_size[1]))
    loc_bb[0] = loc_bb[0]*conv_y
    loc_bb[1] = loc_bb[1]*conv_x
    loc_bb[2] = max(loc_bb[2]*conv_x, 0)
    loc_bb[3] = max(loc_bb[3]*conv_y, 0)
    return [int(value) for value in loc_bb]

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

    return np.stack([convert_bb_from_json(bb_json[f.split("/")[-1]], desired_size, s) for f,s in zip(filenames, sizes)],).astype(np.float32)
