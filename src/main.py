import os
from tqdm import tqdm
from PIL import Image
import numpy as np


def create_SemMaks(path, fold, dest_path):
    images = np.load(os.path.join(path, fold, 'images.npy'))
    print('Loaded {} images'.format(len(images)))
    organs = np.load(os.path.join(path, fold, 'types.npy'))
    masks = np.load(os.path.join(path, fold, 'masks.npy'))
    print('Loaded {} masks'.format(len(masks)))

    assert len(images) == len(organs) == len(masks)

    count = 0
    for image, organ, mask in tqdm(zip(images, organs, masks), total=len(images)):
        img = Image.fromarray(image.astype(np.uint8))
        img.save(os.path.join(dest_path, 'images/{}_{}_{}.png'.format(fold, organ, count)))

        raw_mask = mask[:, :, :]
        sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
        # swaping channels 0 and 5 so that BG is at 0th channel
        sem_mask = np.where(sem_mask == 5, 6, sem_mask)
        sem_mask = np.where(sem_mask == 0, 5, sem_mask)
        sem_mask = np.where(sem_mask == 6, 0, sem_mask)

        semMask = Image.fromarray(sem_mask)
        semMask.save(os.path.join(dest_path, 'seg_masks/{}_{}_{}.png'.format(fold, organ, count)))

        count += 1

create_SemMaks(path='data/pannuke', fold='fold3', dest_path='data/pannuke')
