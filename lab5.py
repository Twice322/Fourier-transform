import numpy as np
import cv2
import pylab
import matplotlib.pyplot as plt

N = 7
t = np.linspace(0, 440, 440)
y_1 = np.cos(0.5 * N * t) + (N * np.sin(t + (2*N + 1)*t))
y_2 = [0.05 * N if abs(i) <= 10 * N else 0 for i in t]

# Функция центрирования
def to_center_func(array_of_values, centered=True):
    centered_image = []
    for i in range(len(array_of_values)):
        if (centered):
            centered_image.append(array_of_values[i] * ((-1) ** i))
        else:
            centered_image.append(array_of_values[i].real * ((-1) ** i))
    return np.array(centered_image)

# Прямое преобразование Фурье
def fourier_transformation_forward(centered_image):
    centered_image = centered_image.copy()
    empty_complex_array = np.zeros(len(centered_image), dtype=complex)
    centered_image_length = len(centered_image)
    for k in range(centered_image_length):
        for n in range(centered_image_length):
            empty_complex_array[k] += centered_image[n] * np.exp(-2 * np.pi * 1j * k * n / centered_image_length)
    return np.array(empty_complex_array)

# Обратное преобразование Фурье
def fourier_transformation_backward(filtered_array):
    empty_complex_array = np.zeros(len(filtered_array), dtype=complex)
    filtered_array_length = len(filtered_array)
    for k in range(filtered_array_length):
        for n in range(filtered_array_length):
            empty_complex_array[k] += filtered_array[n] * np.exp(2 * np.pi * 1j * k * n / filtered_array_length)
        empty_complex_array[k] *= 1 / filtered_array_length
    return np.array(empty_complex_array)

# Фильтр низких частот
def filter_fourier_lf(transformed_array, shift):
    transformed_array = transformed_array.copy()
    transformed_array_length = transformed_array.size
    transformed_array[:transformed_array_length//2 - shift] = transformed_array[
                                                              :transformed_array_length//2 - shift] * 0
    transformed_array[transformed_array_length//2 + shift + 1:] = transformed_array[
                                                                  transformed_array_length//2 + shift + 1:] * 0
    return transformed_array

# Фильтр высоких частот
def filter_fourier_hf(filtered_array, shift):
    filtered_array = filtered_array.copy()
    filtered_array_length = len(filtered_array)
    filtered_array[filtered_array_length // 2 - shift:
                   filtered_array_length // 2 + shift + 1] = 0 * filtered_array[filtered_array_length // 2- shift:
                                                                                filtered_array_length // 2 + shift + 1]
    return np.array(filtered_array)


# Полосный фильтр
def filter_fourier_pf(filtered_array, shift, freq_offset):
    filtered_array = filtered_array.copy()
    center = len(filtered_array) // 2
    filtered_array[:center - shift - freq_offset] *= 0
    filtered_array[center - shift + freq_offset + 1: center] *= 0
    filtered_array[center: center + shift - freq_offset] *= 0
    filtered_array[center + shift + freq_offset + 1:] *= 0
    return np.array(filtered_array)

# Функция для вызова фильтрации для одномерной последовательности
def main(y, number, shift_array):
    shift_lf, shift_hf, shift_pf = shift_array
    centered_image = to_center_func(y)
    transformed_image_forward = fourier_transformation_forward(centered_image.copy())

    filtered_image_lf = filter_fourier_lf(transformed_image_forward.copy(), shift_lf)
    transformed_image_backward = fourier_transformation_backward(filtered_image_lf.copy())
    centered_again_image_lf = to_center_func(transformed_image_backward.copy(), False)

    filtered_image_hf = filter_fourier_hf(transformed_image_forward.copy(), shift_hf)
    transformed_image_backward = fourier_transformation_backward(filtered_image_hf.copy())
    centered_again_image_hf = to_center_func(transformed_image_backward)

    filtered_image_pf = filter_fourier_pf(transformed_image_forward.copy(), shift_pf[0], shift_pf[1])
    transformed_image_backward = fourier_transformation_backward(filtered_image_pf)
    centered_again_image_pf = to_center_func(transformed_image_backward)


    pylab.subplot(3, 1, 1)
    pylab.plot(t, y, alpha=1)
    pylab.plot(centered_again_image_lf, alpha=1)
    pylab.grid()
    pylab.title(f'Низко-частотный фильтр для {number}-ой последовательности')
    pylab.subplot(3, 1, 2)
    pylab.plot(t, y, alpha=1)
    pylab.plot(centered_again_image_hf , alpha=1)
    pylab.grid()
    pylab.title(f'Высоко-частотный фильтр для {number}-ой последовательности')
    pylab.subplot(3, 1, 3)
    pylab.plot(t, y, alpha=1)
    pylab.plot(centered_again_image_pf, alpha=1)
    pylab.grid()
    pylab.title(f'Полосный фильтр для {number}-ой последовательности')
    plt.show()

main(y_1, 1, [195, 195, (193, 1)])
main(y_2, 2, [50, 50, (193, 10)])



image = cv2.imread('test1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Функция центрирования двумерной последовательности
def to_2d_center_func(input_image, centered=True):
    (image_height, image_width) = np.shape(input_image)
    centered_image = np.zeros(shape=(image_height, image_width))
    for i in range(image_height):
        empty_array = []
        for j in range(image_width):
            if (centered):
                empty_array.append(input_image[i][j] * ((-1) ** (i + j)))
            else:
                empty_array.append(input_image[i][j].real * ((-1) ** (i + j)))
        centered_image[i] = empty_array
    return np.array(centered_image)

# Функция прямого преобразования Фурье для двумерного преобразования
def fourier_transform_forward(centered_image):
    centered_image = centered_image.copy()
    (image_height, image_width) = np.shape(centered_image)
    empty_array = np.zeros(shape=(image_height, image_width), dtype=np.complex128)
    for k in range(image_height):
        for l in range(image_width):
            for m in range(image_height):
                for n in range(image_width):
                    empty_array[k][l] += (centered_image[m][n] * (np.exp(-2 * np.pi * 1j * k * m / image_height))
                                          * (np.exp(-2 * np.pi * 1j * l * n / image_width)))
    return np.array(empty_array)

# Функция обратного преобразования Фурье для двумерного преобразования
def fourier_transform_backward(transformed_image):
    (image_height, image_width) = np.shape(transformed_image)
    empty_complex_array = np.zeros(shape=(image_height, image_width), dtype=complex)
    for k in range(image_height):
        for l in range(image_width):
            for m in range(image_height):
                for n in range(image_width):
                    empty_complex_array[k][l] += (transformed_image[m][n]
                    * (np.exp(2 * np.pi * 1j * k * m / image_height)) * \
                    (np.exp(2 * np.pi * 1j * l * n / image_width)))
            empty_complex_array[k][l] *= 1/(image_height * image_width)
    return np.array(empty_complex_array)

# Фильтр низких частот для двумерной последовательности
def filter_lf_2d(transformed_image, shift):
    transformed_image = transformed_image.copy()
    array_height, array_width = np.shape(transformed_image)
    transformed_image[:array_height // 2 - shift[0], :] *= 0
    transformed_image[array_height // 2 + shift[0] + 1:, :] *= 0
    transformed_image[:, :array_width // 2 - shift[1]] *= 0
    transformed_image[:, array_width // 2 + shift[1] + 1:] *= 0
    return np.array(transformed_image)

# Фильтр высоких частот для двумерной последовательности
def filter_hf_2d(transformed_image, shift):
    transformed_image = transformed_image.copy()
    array_height, array_width = np.shape(transformed_image)
    transformed_image[array_height // 2 - shift[0]: array_height//2 + shift[0] + 1,
        array_width // 2 - shift[1]: array_width // 2 + shift[1] + 1
    ] *= 0

    return np.array(transformed_image)

# Полосный для двумерной последовательности
def filter_pr_2d(transformed_image, *args):
    cut_freq, shift = args
    transformed_image = transformed_image.copy()
    transformed_image[: cut_freq[0] - shift[0], : ] *= 0
    transformed_image[cut_freq[0] + shift[0] + 1:, :] *= 0
    transformed_image[:, :cut_freq[1] - shift[1]] *= 0
    transformed_image[:, :cut_freq[1] + shift[1] + 1] *= 0
    return np.array(transformed_image)

# Алгоритм фильтрации
def to_filter_image(filter_name, *args):
    start_image = cv2.imread('patrik.jpg')
    gray_start_image = cv2.cvtColor(start_image, cv2.COLOR_BGR2GRAY)
    center_image = to_2d_center_func(gray_start_image.copy())
    fourier_transformed_image_forward = np.fft.fft2(center_image.copy())
    filtered_image = filter_name(fourier_transformed_image_forward.copy(), *args)
    fourier_transformed_image_backward = np.fft.ifft2(filtered_image)
    return to_2d_center_func(fourier_transformed_image_backward, False)


self_centred_image = to_2d_center_func(gray_image)
self_fourier_transformed_image_forward = fourier_transform_forward(self_centred_image)
numpy_fourier_transformed_image_forward = np.fft.fft2(self_centred_image)
self_fourier_transformed_image_backward = fourier_transform_backward(self_fourier_transformed_image_forward)
numpy_fourier_transformed_image_backward = np.fft.ifft2(self_fourier_transformed_image_forward)


pylab.subplot(2, 2, 1)
pylab.imshow(np.abs(self_fourier_transformed_image_forward), cmap='gray')
pylab.title(f'Собственный алгоритм двухмерного преобразования Фурье')
pylab.subplot(2, 2, 2)
pylab.imshow(np.abs(numpy_fourier_transformed_image_forward), cmap='gray')
pylab.title(f'Алгоритм двухмерного преобразования Фурье при использовании numpy')

pylab.subplot(2, 2, 3)
pylab.imshow(np.abs(self_fourier_transformed_image_backward), cmap='gray')
pylab.subplot(2, 2, 4)
pylab.imshow(np.abs(numpy_fourier_transformed_image_backward), cmap='gray')

pylab.show()


initial_image = cv2.imread('patrik.jpg')
gray_initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)


lf_filter = to_filter_image(filter_lf_2d, (35, 10))
hf_filter = to_filter_image(filter_hf_2d, (55, 20))
ps_filter = to_filter_image(filter_pr_2d, (240, 120), (50, 30))

pylab.subplot(2, 2, 1)
pylab.imshow(gray_initial_image, cmap='gray')
pylab.title('Исходное изображение Патрика')
pylab.subplot(2, 2, 2)
pylab.imshow(np.abs(lf_filter), cmap='gray')
pylab.title('Изображение Патрика при использовании НЧ фильтра')
pylab.subplot(2, 2, 3)
pylab.imshow(np.abs(hf_filter), cmap='gray')
pylab.title('Изображение Патрика при использовании ВЧ фильтра')
pylab.subplot(2, 2, 4)
pylab.imshow(np.abs(ps_filter), cmap='gray')
pylab.title('Изображение Патрика при использовании полосового фильтра')
pylab.show()

