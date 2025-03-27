import cv2
import numpy as np
import time


def video_processing():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)
    i = 0

    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    print("Нажмите 'q', чтобы выйти.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра.")
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)

            if w > 20 and h > 20:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center_x = x + (w // 2)
                center_y = y + (h // 2)

                if i % 5 == 0:
                    print(f"Центр метки: ({center_x}, {center_y})")

                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        cv2.imshow('Отслеживание метки', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_processing()