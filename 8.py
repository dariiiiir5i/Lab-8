import cv2
import time

def video_processing():

    cap = cv2.VideoCapture(0)
    down_points = (640, 480)
    i = 0

    if not cap.isOpened():
        print("Не удалось получить доступ к камере")
        return

    print("Нажмите 'q', чтобы выйти")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка захвата кадра")
                break

            frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(c)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center_x = x + (w // 2)
                center_y = y + (h // 2)

                #Рисование вертикальной и горизонтальной линии через центр метки
                height, width = frame.shape[:2]
                cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)  # Вертикальная линия
                cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)   # Горизонтальная линия

                if i % 5 == 0:
                    print(f"Центр метки: ({center_x}, {center_y})")

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)
            i += 1

    except Exception as e:
        print(f"Произошла ошибка: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video_processing()