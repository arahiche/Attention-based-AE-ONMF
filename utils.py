import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils as kerasutils
import configparser
import os
from skimage.morphology import skeletonize


def SAD(y_true, y_pred):
    """
    SAD loss that compares an input spectra with a reconstructed one
    :param y_true:
    :param y_pred:
    :return : sad
    """
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    A = (y_true * y_pred)
    sad = tf.math.acos(A)
    return sad


def plt_no_margin(img, cmap="viridis"):
    """
    Plots a matrix as an 1-band image figure with no margin and given color map
    :param img:
    :param cmap:
    """

    fig_width, fig_height = img.shape[1] / 100, img.shape[0] / 100  # Figure size inch (ratio fig_size * dpi / 100)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.imshow(img, cmap=cmap)
    plt.axis('off')

    ax.set_position([0, 0, 1, 1])
    plt.show()


def pcc(x, y):
    """
    Computing the Pearson's Correlation Coefficient between x and y \n
    i.e. two 2d matrices (images), of the same size
    :param x:
    :param y:
    :return: pcc
    """

    xi_xm = x - x.mean()
    yi_ym = y - y.mean()

    # formula given in Supplementary materials :
    # A. Rahiche et M. Cheriet
    # Blind Decomposition of Multispectral Document Images Using Orthogonal Nonnegative Matrix Factorization
    # 2021, doi: 10.1109/TIP.2021.3088266.
    r = (np.sum(xi_xm * yi_ym, dtype=np.float64) /
         (np.sqrt(np.sum(xi_xm ** 2, dtype=np.float64)) * np.sqrt(np.sum(yi_ym ** 2, dtype=np.float64))))

    if np.any(np.isnan(r)):
        print("nan in pcc")
        r = np.nan_to_num(r)

    return r


def load_spectrum(path_spectrum):
    """
    Returns a 2D numpy array representing a one-d image (a spectrum), from a path
    :param path_spectrum:
    :return spectrum:
    """
    spectrum = kerasutils.load_img(path_spectrum)
    spectrum = kerasutils.img_to_array(spectrum)
    spectrum = np.array([spectrum])  # Convert single image to a batch.
    spectrum = np.squeeze(spectrum, axis=0)
    spectrum = spectrum[:, :, 0]  # images are "RGB encoded" but the 3 values are equal, so we only keep one of them

    return spectrum


def load_msi(path_msi, n_bands=8):
    """
    Returns a 3D numpy array representing a set of spectra, forming a MultiSpectral Image,
    from a path and a number of bands
    :param path_msi:
    :param n_bands:
    :return msi:
    """

    print(path_msi)
    spectra_path_list = [None] * n_bands
    for i in range(n_bands-1):
        if n_bands == 8:
            # images from MSTex have 8 bands
            spectra_path_list[i] = path_msi + '/F' + str(i+1) + 's.png'
        else:
            # images from MSBin have 11 bands
            spectra_path_list[i] = path_msi + '_' + str(i+1) + '.png'

    first_spectrum = load_spectrum(spectra_path_list[0])
    n_rows = first_spectrum.shape[0]
    n_cols = first_spectrum.shape[1]

    msi = np.empty([n_rows, n_cols, n_bands], dtype=np.float32)

    for i in range(n_bands-1):
        msi[:, :, i] = load_spectrum(spectra_path_list[i])

    if np.any(np.isnan(msi)):
        print("ERROR : NaN in : ", path_msi)

    # normalizing the MSI, usually defined in [0, 255], to  [0, 1] :
    msi = msi / msi.max()
    # note that the normalization is applied using the max of the whole MSI (largest value across each band)

    return msi


def load_config(ini_file_path):
    """
    Extracts config from a .ini file into a python dict, storing parameters about the model such that
    the only variable being passed through successive functions is "params"
    :param ini_file_path:
    :return params:
    """
    config_obj = configparser.ConfigParser()
    config_obj.read(ini_file_path)
    params = dict(config_obj.items("params"))
    for key, value in params.items():
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = params[key]
    return params


def load_data(img_name, params):
    """
    Loads the MSI and the Ground Truth as numpy arrays, and writes some parameters in the dict \n
    from a given image name, whatever the dataset : MSTEx1, MSTEx2, MSBin, train / test
    :param img_name:
    :param params:
    :return data:
    :return params;
    :return gt:
    """
    MSI1_list = ['z30', 'z32', 'z34', 'z35', 'z36', 'z37', 'z38', 'z48', 'z58', 'z59', 'z60', 'z64', 'z67', 'z68',
                 'z70', 'z76', 'z80', 'z82', 'z95', 'z97', 'z100']
    MSI2_list = ['z27', 'z31', 'z43', 'z65', 'z90', 'z582', 'z92', 'z802', 'z822', 'z592']

    # looking for the dataset folder storing the image
    # three folder are assumed to be in the project directory : S_MSI_1, S_MSI_2 and MSBin_v2
    # /!\ these paths might need to be adapted /!\
    if img_name in MSI1_list:
        path_dir = "S_MSI_1/MSI/"
        path_gt = "S_MSI_1/GT/" + img_name + "GT.png"
        n_bands = 8
        dataset = 'MSTEx'
    elif img_name in MSI2_list:
        path_dir = "S_MSI_2/MSI/"
        path_gt = "S_MSI_2/GT/" + img_name + "GT.png"
        n_bands = 8
    else:
        path_dir = "MSBin_v2/data/train/images/"
        MSBin_list = os.listdir(path_dir)
        if img_name + "_0.png" in MSBin_list:
            n_bands = 11
            path_gt = "MSBin_v2/data/train/dibco_labels/fg_1/" + img_name + ".png"
            dataset = 'MSTEx'
        else:
            path_dir = "MSBin_v2/data/test/images/"
            MSBin_list = os.listdir(path_dir)
            if img_name + "_0.png" in MSBin_list:
                n_bands = 11
                path_gt = "MSBin_v2/data/test/dibco_labels/fg_1/" + img_name + ".png"
                dataset = 'MSBin'
            else:
                print(img_name + "not found in MSBin or MS-TEx dataset")
                quit()
    # note that only the main foreground is loaded here even for the MSBin images that might contain a second one

    # loading the MSI
    data = load_msi(path_dir + img_name, n_bands)
    n_bands = data.shape[-1]

    gt = load_spectrum(path_gt)

    gt = gt / gt.max()  # normalizing the gt image in case it is defined as 0-255 instead of 0-1

    print("the loaded MSI shape is : ", data.shape)  # [n1, n2, k]
    print("the loaded GT shape is : ", gt.shape)  # [n1, n2]

    if np.any(np.isnan(data)) or np.any(np.isnan(gt)):
        print(" ERROR : got NaN values while loading data")
        quit()

    num_patches = params["num_patches"]

    # we slightly adjust the number of patches so every batch is full
    # = batch size is constant for every batch, even for the last one
    num_patches = params["batch_size"] * round(num_patches / params["batch_size"])
    print("num_patches : ", num_patches)

    params['num_patches'] = num_patches
    params['n_bands'] = n_bands

    return data, params, gt, dataset


def DRD(abd, gt):

    # DRD: Distance Reciprocal Distortion Metric
    blkSize = 8  # even number
    MaskSize = 5  # odd number

    xm, ym = gt.shape

    # Initialize the padded ground truth matrix
    gt1 = np.zeros((xm + 2, ym + 2), dtype=bool)
    gt1[1:xm + 1, 1:ym + 1] = gt

    # Compute the integral image
    intim = np.cumsum(np.cumsum(gt1, axis=0), axis=1)

    NUBN = 0
    blkSizeSQR = blkSize ** 2

    # Calculate the number of useful block neighbors
    for i in range(2, xm - blkSize + 1, blkSize):
        for j in range(2, ym - blkSize + 1, blkSize):
            blkSum = intim[i + blkSize - 1, j + blkSize - 1] - intim[i - 1, j + blkSize - 1] - intim[
                i + blkSize - 1, j - 1] + intim[i - 1, j - 1]
            if blkSum != 0 and blkSum != blkSizeSQR:
                NUBN += 1

    # Create the weight matrix
    wm = np.zeros((MaskSize, MaskSize))
    ic = jc = (MaskSize + 1) // 2  # center coordinate

    for i in range(MaskSize):
        for j in range(MaskSize):
            wm[i, j] = 1 / np.sqrt((i - ic + 1) ** 2 + (j - jc + 1) ** 2)

    wm[ic - 1, jc - 1] = 0
    wnm = wm / np.sum(wm)  # Normalized weight matrix

    # Resize the matrices with padding
    gt_Resized = np.zeros((xm + ic + 1, ym + jc + 1))
    gt_Resized[ic - 1:xm + ic - 1, jc - 1:ym + jc - 1] = gt

    abd_Resized = np.zeros((xm + ic + 1, ym + jc + 1))
    abd_Resized[ic - 1:xm + ic - 1, jc - 1:ym + jc - 1] = abd

    temp_fp_Resized = np.logical_and(abd_Resized == 0, gt_Resized != 0)
    temp_fn_Resized = np.logical_and(abd_Resized != 0, gt_Resized == 0)
    Diff = np.logical_or(temp_fp_Resized, temp_fn_Resized)

    xm2, ym2 = Diff.shape
    SumDRDk = 0

    # Compute the DRD metric
    for i in range(ic - 1, xm2 - ic + 1):
        for j in range(jc - 1, ym2 - jc + 1):
            if Diff[i, j]:
                Local_Diff = np.logical_xor(gt_Resized[i - ic + 1:i + ic, j - ic + 1:j + ic], abd_Resized[i, j])
                DRDk = np.sum(Local_Diff * wnm)
                SumDRDk += DRDk

    return SumDRDk / NUBN


def PSNR(gt, FP, FN):
    xm, ym = gt.shape
    err = np.sum(FP | FN) / (xm * ym)
    psnr = 10 * np.log10(1 / err)
    if np.isinf(psnr):
        psnr = 0
    return psnr


def pFM(abd, gt, p):

    gt_inverted = np.logical_not(gt)  # Invert the binary img
    SKL_GT_inverted = skeletonize(gt_inverted)  # skeletonize
    SKL_GT = np.logical_not(SKL_GT_inverted)  # Invert again

    # True Positive (TP): we predict a label of 0 (text), and the ground truth is 0.
    SKL_TP = np.sum(np.logical_and(abd == 0, SKL_GT == 0))

    # False Negative (FN): we predict a label of 1 (background), but the ground truth is 0 (text).
    SKL_FN = np.sum(np.logical_and(abd == 1, SKL_GT == 0))

    pseudo_r = SKL_TP / (SKL_FN + SKL_TP)

    return 100 * 2 * (p * pseudo_r) / (p + pseudo_r)


def compute_scores(abd, gt):
    # True Positive (TP): we predict a label of 0 (text), and the ground truth is 0.
    TP = np.sum(np.logical_and(abd == 0, gt == 0))

    # True Negative (TN): we predict a label of 1 (background), and the ground truth is 1.
    TN = np.sum(np.logical_and(abd == 1, gt == 1))

    # False Positive (FP): we predict a label of 0 (text), but the ground truth is 1 (background).
    FP = np.sum(np.logical_and(abd == 0, gt == 1))

    # False Negative (FN): we predict a label of 1 (background), but the ground truth is 0 (text).
    FN = np.sum(np.logical_and(abd == 1, gt == 0))

    # print("TP : ", TP)
    # print("TN : ", TN)
    # print("FP : ", FP)
    # print("FN : ", FN)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    FM = 2 * recall * precision / (recall + precision)

    drd = DRD(abd, gt)

    NRfn = FN / (FN + TP)
    NRfp = FP / (FP + TN)
    NRM = (NRfn + NRfp) / 2

    ACC = (TP + TN) / (TP + TN + FP + FN)

    psnr = PSNR(gt=gt, FP=FP, FN=FN)

    pfm = pFM(abd, gt, precision)

    # print("FM %: ", FM*100)
    # print("DRD ?: ", drd)
    # print("NRM ×10−2: ", NRM*100)
    # print("ACC %: ", ACC*100)
    # print("PSNR %: ", psnr)
    # print("pFM %: ", pfm)

    return np.array([FM, drd, NRM, ACC, psnr, pfm])
