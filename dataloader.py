from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
import numpy as np


def load_tractography_derivatives_3D():
    evec1, a, b = load_nifti('data/Preprocessed/DTIfit_output/dti_V1.nii.gz', return_img=True)
    evec2, a, b = load_nifti('data/Preprocessed/DTIfit_output/dti_V2.nii.gz', return_img=True)
    evec3, a, b = load_nifti('data/Preprocessed/DTIfit_output/dti_V3.nii.gz', return_img=True)

    eval1, a, b = load_nifti('data/Preprocessed/DTIfit_output/dti_L1.nii.gz', return_img=True)
    eval2, a, b = load_nifti('data/Preprocessed/DTIfit_output/dti_L2.nii.gz', return_img=True)
    eval3, a, b = load_nifti('data/Preprocessed/DTIfit_output/dti_L3.nii.gz', return_img=True)

    evec1 = np.array(evec1)
    evec2 = np.array(evec2)
    evec3 = np.array(evec3)

    eval1 = np.array(eval1)
    eval2 = np.array(eval2)
    eval3 = np.array(eval3)

    max_directions = np.argmax(np.stack((eval1, eval2, eval3), axis=2), axis=2)
    stackedEvals = np.stack((eval1, eval2, eval3), axis=3)
    stackedEvecs = np.stack((evec1, evec2, evec3), axis=3)
    directions = np.array([[[stackedEvals[i, j, k][max_directions[i, j, k]] *
        stackedEvecs[i, j, k][max_directions[i, j, k]]
        for k in range(len(max_directions[0, 0]))]
        for j in range(len(max_directions[0]))]
        for i in range(len(max_directions))])

    return directions

def load_fibercup_tractography_derivatives_3D():
    niftipath = 'data\\fibercup\\R3.nii.gz'
    bvalspath = 'data\\fibercup\\R3.bvals'
    bvecspath = 'data\\fibercup\\R3.bvecs'

    print("Fetching data")
    data, affine = load_nifti(niftipath)

    bvals, bvecs = read_bvals_bvecs(bvalspath, bvecspath)

    gtab = gradient_table(bvals, bvecs)

    print("Fitting tensor model...")
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    quadratic_form = tenfit.quadratic_form

    evals, evecs = np.linalg.eig(quadratic_form)

    max_directions = np.argmax(evals, axis=2)
    directions = np.array([[[evals[i, j, k][max_directions[i, j, k]] *
        evecs[i, j, k][max_directions[i, j, k]]
        for k in range(len(max_directions[0, 0]))]
        for j in range(len(max_directions[0]))]
        for i in range(len(max_directions))])

    return np.real(directions)

def load_fibercup_tractography_derivatives_2D():
    niftipath = 'data\\fibercup\\R3.nii.gz'
    bvalspath = 'data\\fibercup\\R3.bvals'
    bvecspath = 'data\\fibercup\\R3.bvecs'

    print("Fetching data")
    data, affine = load_nifti(niftipath)

    bvals, bvecs = read_bvals_bvecs(bvalspath, bvecspath)

    gtab = gradient_table(bvals, bvecs)

    print("Fitting tensor model...")
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    quadratic_form = tenfit.quadratic_form

    evals, evecs = np.linalg.eig(quadratic_form)

    max_directions = np.argmax(evals, axis=2)
    directions = np.array([[[evals[i, j, k][max_directions[i, j, k]] *
        evecs[i, j, k][max_directions[i, j, k]]
        for k in range(len(max_directions[0, 0]))]
        for j in range(len(max_directions[0]))]
        for i in range(len(max_directions))])

    return np.real(directions[:, :, 1, :2])