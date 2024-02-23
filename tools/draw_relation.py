import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


def find_rel_id(anno, subject_id, object_id):
    rel_ids = []
    for idx, a in enumerate(anno):
        if a['object']['category'] == object_id and \
                a['subject']['category'] == subject_id:
            rel_ids.append(idx)
    return rel_ids


def draw_relation_bbox(file_name, annotation, predicates, objects, rel_id=None):
    # file_name = list(annotation.keys())[0]
    fig, ax = plt.subplots(1)
    img = mpimg.imread('../resource/VAD/sg_dataset/sg_dataset/sg_test_images/' + file_name)
    # Display the image
    ax.imshow(img)

    if type(rel_id) == int:
        rel_id = [rel_id]

    for i in range(0, len(annotation[file_name])):
        relation = predicates[annotation[file_name][i]['predicate']]
        object = objects[annotation[file_name][i]['object']['category']]
        object_coord = annotation[file_name][i]['object']['bbox']
        subject = objects[annotation[file_name][i]['subject']['category']]
        subject_coord = annotation[file_name][i]['subject']['bbox']

        # print(f'{i}, {relation}, {subject}, {object}')
        # print('-' * 25)

        if rel_id is not None and i not in rel_id:
            continue

        print(f'Ploted relationship: {subject}, {relation}, {object}')
        rect1 = patches.Rectangle(
            (object_coord[2], object_coord[0]),
            object_coord[3] - object_coord[2], object_coord[1] - object_coord[0],
            linewidth=5, edgecolor=[25 / 255, 64 / 255, 117 / 255, 1], facecolor='none')
        rect2 = patches.Rectangle(
            (subject_coord[2], subject_coord[0]),
            subject_coord[3] - subject_coord[2], subject_coord[1] - subject_coord[0],
            linewidth=5, edgecolor=[227 / 255, 135 / 255, 66 / 255, 1], facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    plt.axis('off')
    plt.show()
    fig.savefig('plot.pdf', bbox='tight')
