import h5py
import os
import pickle
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.data import DatasetAndSupport, get_item, sample_to_sample_plus
# from evaluate.standard_metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.metrics import jaccard_index, chamfer_weighted_symmetric
from utils.utils_common import crop, DataModes


class Sample:
    def __init__(self, x, y, atlas):
        self.x = x
        self.y = y
        self.atlas = atlas


class FissuresDataset(Dataset):

    def __init__(self, data, cfg, mode):
        self.data = data

        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return get_item(item, self.mode, self.cfg)


class Fissures(DatasetAndSupport):

    def quick_load_data(self, cfg, trial_id):

        data_root = cfg.dataset_path
        data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            with open(data_root + '/pre_computed_data_' + datamode + '_fold' + str(trial_id) + '.pickle', 'rb') as handle:
                samples = pickle.load(handle)
                new_samples = sample_to_sample_plus(samples, cfg, datamode)
                data[datamode] = FissuresDataset(new_samples, cfg, datamode)

        return data

    def pre_process_dataset(self, cfg):
        '''
         :
        '''
        print('Data pre-processing - Fissures Dataset')

        data_root = cfg.dataset_path
        down_sample_shape = cfg.patch_shape  # (64, 64, 64)  TODO: less downsampling!
        largest_image_size = cfg.largest_image_shape  # (352, 352, 352)

        ids = sorted(dir for dir in os.listdir('{}/imagesTr'.format(data_root)))

        print('Load data...')

        inputs = []
        labels = []

        vals = []
        sizes = []
        for itr, sample in enumerate(ids):
            if '.nii.gz' in sample:
                print(sample)
                x, y = self.read_sample(data_root, sample, down_sample_shape, largest_image_size)
                inputs += [x.cpu()]
                labels += [y.cpu()]

        inputs_ = [i[None].data.numpy() for i in inputs]
        labels_ = [i[None].data.numpy() for i in labels]
        inputs_ = np.concatenate(inputs_, axis=0)
        labels_ = np.concatenate(labels_, axis=0)

        hf = h5py.File(data_root + '/data.h5', 'w')
        hf.create_dataset('inputs', data=inputs_)
        hf.create_dataset('labels', data=labels_)
        hf.close()

        print('Saving data...')

        data = {}
        # down_sample_shape = (32, 128, 128)
        for fold, split in enumerate(cfg.split):
            print(f'Fold {fold}')
            for i, (datamode, split_ids) in enumerate(zip([DataModes.TRAINING, DataModes.TESTING], split.values())):

                indices = [j for j, file in enumerate(ids) if any(sid in file for sid in split_ids)]

                samples = []

                for j in indices:
                    x = inputs[j]
                    y = labels[j]

                    samples.append(Sample(x, y, None))

                with open(data_root + '/pre_computed_data_' + datamode + '_fold' + str(fold) + '.pickle', 'wb') as handle:
                    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('\nPre-processing complete')
        return data

    def evaluate(self, target, pred, cfg):
        results = {}

        if target.voxel is not None:
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

        if target.points is not None:
            target_points = target.points
            pred_points = pred.mesh
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(),
                                                                               pred_points[i]['vertices'])

            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        if 'chamfer_weighted_symmetric' in new_value[DataModes.TESTING]:
            key = 'chamfer_weighted_symmetric'
            new_value = new_value[DataModes.TESTING][key]

            if best_so_far is None:
                return True
            else:
                best_so_far = best_so_far[DataModes.TESTING][key]
                return True if np.mean(new_value) < np.mean(best_so_far) else False

        elif 'jaccard' in new_value[DataModes.TESTING]:
            key = 'jaccard'
            new_value = new_value[DataModes.TESTING][key]

            if best_so_far is None:
                return True
            else:
                best_so_far = best_so_far[DataModes.TESTING][key]
                return True if np.mean(new_value) > np.mean(best_so_far) else False

    def read_sample(self, data_root, sample, out_shape, pad_shape):
        img = resample_equal_spacing(sitk.ReadImage('{}/imagesTr/{}'.format(data_root, sample)))
        label = resample_equal_spacing(sitk.ReadImage('{}/labelsTr/{}'.format(data_root, sample.replace('_0000', ''))), use_nearest_neighbor=True)

        x = sitk.GetArrayFromImage(img)
        y = sitk.GetArrayFromImage(label)
        labelsum = y.sum()

        D, H, W = x.shape
        center_z, center_y, center_x = D // 2, H // 2, W // 2
        D, H, W = pad_shape
        x = crop(x, (D, H, W), (center_z, center_y, center_x))
        y = crop(y, (D, H, W), (center_z, center_y, center_x))

        assert labelsum == y.sum(), f'fissures have been cropped out'

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x = F.interpolate(x[None, None], out_shape, mode='trilinear', align_corners=False)[0, 0]
        y = F.interpolate(y[None, None].float(), out_shape, mode='nearest')[0, 0].long()

        return x, y


def resample_equal_spacing(img: sitk.Image, target_spacing: float = 1., use_nearest_neighbor: bool = False):
    """ resmample an image so that all dimensions have equal spacing

    :param img: input image to resample
    :param target_spacing: the desired spacing for all 3 dimensions in the output
    :return: resampled image
    """
    output_size = [round(size * (spacing/target_spacing)) for size, spacing in zip(img.GetSize(), img.GetSpacing())]
    return sitk.Resample(img, size=list(output_size), outputSpacing=(1, 1, 1),
                         interpolator=sitk.sitkNearestNeighbor if use_nearest_neighbor else sitk.sitkLinear)

