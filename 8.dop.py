import cv2
import numpy as np
import time

def overlay_fly(frame, fly, center_x, center_y):
    """
    Наложение изображения мухи с прозрачностью на кадр
    """
    fly_height, fly_width = fly.shape[:2]

    # Расчет координат для размещения мухи
    start_x = max(center_x - fly_width // 2, 0)
    start_y = max(center_y - fly_height // 2, 0)
    end_x = min(center_x + fly_width // 2, frame.shape[1])
    end_y = min(center_y + fly_height // 2, frame.shape[0])

    # Проверка на валидность размеров
    overlay_fly_width = end_x - start_x
    overlay_fly_height = end_y - start_y

    if overlay_fly_width <= 0 or overlay_fly_height <= 0:
        return

    # Работа с прозрачностью
    alpha_fly = fly[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_fly

    # Наложение с учетом прозрачности для каждого цветового канала
    for c in range(0, 3):
        frame[start_y:end_y, start_x:end_x, c] = (
            alpha_fly * fly[0:overlay_fly_height, 0:overlay_fly_width, c] +
            alpha_frame * frame[start_y:end_y, start_x:end_x, c]
        )

def video_processing():
    """
    Основная функция обработки видео с камеры
    """
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)  # Размер кадра для обработки
    i = 0  # Счетчик кадров

    # Загрузка изображения мухи
    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    if fly is None:
        print("Ошибка: не удалось загрузить изображение мухи (fly64.png)")
        print("Убедитесь, что файл находится в той же папке, что и скрипт")
        return

    fly = cv2.resize(fly, (64, 64), interpolation=cv2.INTER_LINEAR)

    # Проверка открытия камеры
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    print("Запуск обработки видео...")
    print("Нажмите 'q' для выхода")

    while True:
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр с камеры")
            break

        # Изменение размера кадра
        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)

        # Преобразование в оттенки серого и размытие
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Бинаризация для выделения метки
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Выбор самого большого контура
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Фильтрация по размеру (игнорирование мелких объектов)
            if w > 20 and h > 20:
                # Рисование прямоугольника вокруг метки
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Вычисление центра метки
                center_x = x + (w // 2)
                center_y = y + (h // 2)

                # Вывод координат центра (каждый 5-й кадр)
                if i % 5 == 0:
                    print(f"Центр метки: ({center_x}, {center_y})")

                # Наложение изображения мухи
                overlay_fly(frame, fly, center_x, center_y)

                # Рисование точки в центре
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Отображение результата
        cv2.imshow('Отслеживание метки', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Небольшая задержка для уменьшения нагрузки
        time.sleep(0.1)
        i += 1

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_processing()