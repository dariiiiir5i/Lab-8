import cv2
import numpy as np

# Загрузка изображения
image_path = "variant-8.jpg"
image = cv2.imread(image_path)

# Проверка успешной загрузки
if image is None:
    print(f"Ошибка: не удалось загрузить изображение {image_path}!")
    exit()

# Получение размеров изображения
height, width, _ = image.shape

# Вычисление центра изображения
x_center = width // 2
y_center = height // 2

# Определение области обрезки (400x400 пикселей вокруг центра)
size = 200  # Половина размера конечного изображения
x_start = x_center - size
y_start = y_center - size
x_end = x_center + size
y_end = y_center + size

# Проверка, чтобы координаты не вышли за границы изображения
x_start = max(0, x_start)
y_start = max(0, y_start)
x_end = min(width, x_end)
y_end = min(height, y_end)

# Обрезка изображения
cropped_image = image[y_start:y_end, x_start:x_end]

# Создание изображения с визуализацией процесса
visualization = image.copy()

# 1. Рисуем центральную точку
cv2.circle(visualization, (x_center, y_center), 5, (0, 0, 255), -1)

# 2. Рисуем прямоугольник области обрезки
cv2.rectangle(visualization, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

# 3. Добавляем текстовую информацию
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(visualization, f"Original: {width}x{height}", (10, 30),
            font, 0.7, (255, 255, 255), 2)
cv2.putText(visualization, f"Crop Area: {x_end-x_start}x{y_end-y_start}", (10, 60),
            font, 0.7, (255, 255, 255), 2)
cv2.putText(visualization, "Green rectangle - crop area", (10, 90),
            font, 0.7, (0, 255, 0), 2)
cv2.putText(visualization, "Red dot - image center", (10, 120),
            font, 0.7, (0, 0, 255), 2)

# Собираем все изображения в одно окно
top_row = np.hstack([image, visualization])
bottom_row = cv2.resize(cropped_image, (width, y_end-y_start))  # Масштабируем обрезанное изображение
if bottom_row.shape[1] < top_row.shape[1]:
    # Добавляем черную полосу, если изображения разной ширины
    border_size = top_row.shape[1] - bottom_row.shape[1]
    bottom_row = cv2.copyMakeBorder(bottom_row, 0, 0, 0, border_size,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])

final_image = np.vstack([top_row, bottom_row])

# Сохранение результата
output_path = "variant-8-cropped.jpg"
cv2.imwrite(output_path, cropped_image)

# Отображение результата
cv2.imshow("Image Cropping Process", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Изображение успешно обрезано и сохранено как '{output_path}'")
print(f"Размер исходного изображения: {width}x{height}")
print(f"Размер обрезанного изображения: {x_end-x_start}x{y_end-y_start}")