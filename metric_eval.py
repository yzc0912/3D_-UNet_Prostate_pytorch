import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.ndimage._ni_support import _normalize_sequence




def dc(result, reference):
    result = np.atleast_1d(result.astype(bool))  # Change np.bool to bool
    reference = np.atleast_1d(reference.astype(bool))  # Change np.bool to bool
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def jc(result, reference):
    result = np.atleast_1d(result.astype(bool))  # Change np.bool to bool
    reference = np.atleast_1d(reference.astype(bool))  # Change np.bool to bool
    
    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    
    jc = float(intersection) / float(union)
    
    return jc


def hd(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def hd95(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    计算result和reference二值对象的表面距离。
    该函数计算的是result中表面像素到reference中最近表面像素的距离。

    参数
    ----------
    result : array_like
        输入数据，包含二值对象。
    reference : array_like
        输入数据，包含二值对象。
    voxelspacing : float 或 sequence of float, 可选
        各维度的体素间距。默认为None，即每个维度的间距为1。
    connectivity : int, 可选
        邻域/连接性，用于确定二值对象的表面。通常为1或2。默认为1。

    返回
    -------
    sds : ndarray
        结果是一个包含表面距离的数组，其中每个元素是result表面像素到reference表面像素的距离。

    异常
    -----
    如果任何输入图像为空，抛出RuntimeError。
    """
    # 确保输入为布尔型数组
    result = np.atleast_1d(result.astype(bool))  # Change np.bool to bool
    reference = np.atleast_1d(reference.astype(bool))  # Change np.bool to bool

    # 如果有体素间距，则规范化它
    if voxelspacing is not None:
        voxelspacing = _normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # 创建二值结构，用于表面计算
    footprint = generate_binary_structure(result.ndim, connectivity)

    # 检查输入图像是否为空
    if np.count_nonzero(result) == 0:
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if np.count_nonzero(reference) == 0:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # 提取表面：通过二值腐蚀获得物体的边界
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # 计算表面距离：距离变换计算到参考表面的距离
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]  # 只计算表面像素的距离

    return sds

