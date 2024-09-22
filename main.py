import keras.layers
from model import *
from sklearn.feature_extraction.image import extract_patches_2d
from howe import howe_alg

# model parameters are set in the config.ini file
params = load_config("config.ini")

# loading the image from its name, works for MSTEx and MSBin
img_name = 'BT51'
data, params, gt, dataset = load_data(img_name=img_name, params=params)

# patches are extracted for each run, randomly taken from data
patches = extract_patches_2d(data, patch_size=(params["patch_size"], params["patch_size"]),
                             max_patches=params["num_patches"])

# the Autoencoder is defined then trained
autoencoder = Autoencoder(params)
autoencoder.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=params["learning_rate"],
                                                                 decay=0.0),
                    loss=keras.losses.MeanSquaredError())
autoencoder.train(patches=patches)

# Abundance maps can be obtained
abundances = autoencoder.getAbundances(data)
# abundances shape is [n1, n2, k] where n1 n2 are the image dimensions, k is the number of elements

# Reconstructed MSI can be obtained
# output = autoencoder.getOutput(abundances)

# Howe binarization is applied on the abundance maps
howe_binarized = np.empty_like(abundances)
for k in range(0, abundances.shape[-1]):
    print("Applying Howe Binarization Algorithm on extracted image n°" + str(k))
    howe_binarized[:, :, k] = howe_alg(1 - abundances[:, :, k])

# "unknown" MSBin areas should not be considered for the evaluation
if dataset == 'MSBin':
    blue_gt_path = 'MSBin_v2/data/test/labels/' + img_name + ".png"  # /!\ path might need to be adapted /!\
    blue_gt_img = np.squeeze(np.array([kerasutils.img_to_array(kerasutils.load_img(blue_gt_path))]), axis=0)

    blue_gt = (blue_gt_img == [0, 0, 255]).all(axis=-1).astype(int)
    blue_gt_extended = np.repeat(blue_gt[:, :, np.newaxis], howe_binarized.shape[-1], axis=2)
    howe_binarized[blue_gt_extended == 1] = 1

# scores can be computed by comparing the binarized abundance maps with the ground truth
scores = np.empty([params["num_endmembers"], 6])
for k in range(params["num_endmembers"]):
    print("computing score for image " + str(k) + " vs gt")
    scores[k] = compute_scores(howe_binarized[:, :, k], gt)  # np.array([FM, drd, NRM, ACC, psnr, pfm])

# images and scores can be saved
# np.savez_compressed(img_name + '/abd_r' + str(i) + '.npz', abundances)
# np.savez_compressed(img_name + '/howe_binzd_r' + str(i) + '.npz', howe_binarized)
# np.savez_compressed(img_name + '/scores_r' + str(i) + '.npz', scores)
