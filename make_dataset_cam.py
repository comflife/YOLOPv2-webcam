import cv2
import os
import time
import glob

def get_latest_file_number(path):
    # 폴더에서 .jpg 파일 목록을 가져옴
    files = glob.glob(os.path.join(path, '*.jpg'))

    if not files: # 파일이 없는 경우 0 반환
        return 0

    # 파일이름에서 숫자 부분만 추출하고 정수로 변환
    numbers = [int(f.replace(path+'/', '').replace('.jpg', '')) for f in files]
    
    # 숫자 목록에서 가장 큰 숫자를 반환
    return max(numbers)

def main():
    # 사용할 웹캠 설정
    cap = cv2.VideoCapture(2)

    # 저장할 경로 설정
    path = '/home/bg/myssd/kcity_image/images2'
    os.makedirs(path, exist_ok=True) # 폴더가 없으면 생성

    # 가장 최근의 파일 번호를 가져와 다음 파일 번호를 설정
    next_file_number = get_latest_file_number(path) + 1

    try:
        while True:
            # 웹캠에서 이미지를 캡쳐
            ret, frame = cap.read()

            if not ret:
                break

            # 이미지를 파일에 저장
            filename = os.path.join(path, f'{next_file_number}.jpg')
            cv2.imwrite(filename, frame)

            print(f'Image saved: {filename}')

            # 다음 파일 번호를 설정
            next_file_number += 1

            # 0.5초 대기
            time.sleep(0.2)

    finally:
        # 웹캠 사용 종료
        cap.release()

if __name__ == '__main__':
    main()
