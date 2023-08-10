# ----------------------------------------------------------------------------
# Copyright (C) [2023] Byounggun Park
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

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
