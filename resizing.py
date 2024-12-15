import os
import glob
import numpy as np
import SimpleITK as sitk

def resample_image(image, out_size, is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # 使用内置的 int 类型替代 np.int
    out_size = np.array(out_size, dtype=int)
    
    # 计算新的像素间距
    new_spacing = [ (original_spacing[i] * original_size[i]) / out_size[i] for i in range(3) ]
    
    # 选择插值方法
    if is_label:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear
    
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    
    resampled_image = resample.Execute(image)
    return resampled_image

if __name__ == '__main__':
    new_size = [128, 128, 64]
    
    # 更新文件路径以匹配您的目录结构
    ct_files = sorted(glob.glob('../raw_dataset/test/ct/volume-*.nii'))
    label_files = sorted(glob.glob('../raw_dataset/test/label/segmentation-*.nii'))

    # 设置新的输出目录
    new_ct_dir = 'new_dataset/test/ct'
    new_label_dir = 'new_dataset/test/label'

    # 如果目录不存在，则创建
    os.makedirs(new_ct_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)

    for ct_file, label_file in zip(ct_files, label_files):
        # 处理CT图像
        ct_img = sitk.ReadImage(ct_file)
        reshaped_ct = resample_image(ct_img, new_size, is_label=False)  # 对CT图像使用线性插值

        # 保存重采样后的CT图像
        new_ct_path = os.path.join(new_ct_dir, os.path.basename(ct_file))
        sitk.WriteImage(reshaped_ct, new_ct_path)
        print(f'已重采样并保存CT图像: {new_ct_path}')

        # 处理标签（分割）图像
        label_img = sitk.ReadImage(label_file)
        reshaped_label = resample_image(label_img, new_size, is_label=True)  # 对标签图像使用最近邻插值

        # 保存重采样后的标签图像
        new_label_path = os.path.join(new_label_dir, os.path.basename(label_file))
        sitk.WriteImage(reshaped_label, new_label_path)
        print(f'已重采样并保存标签图像: {new_label_path}')

    print("\n处理完成！")
