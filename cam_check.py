import cv2

def calculate_aspect_ratio(frame_width, frame_height):
    aspect_ratio = frame_width / frame_height
    return aspect_ratio

def main():
    # USB 카메라를 사용하기 위해 카메라 인덱스를 설정합니다.
    camera_index = 0  # 일반적으로 0부터 시작하여 증가시키면 다른 카메라를 선택할 수 있습니다.

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Unable to access the camera with index {camera_index}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 영상 처리 등 추가 작업을 원한다면 여기에 추가

        aspect_ratio = calculate_aspect_ratio(frame_width, frame_height)

        # 비율 출력
        print(f"The aspect ratio of the camera stream is: {aspect_ratio:.2f}")

        cv2.imshow("Camera Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
