import cv2
import pytesseract


def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"./preprocess/img_gray.png", img)
    return img


def blur(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(r"./preprocess/img_blur.png", img)
    return img_blur


def threshold(img):
    # pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.threshold(img, 20, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    cv2.imwrite(r"./preprocess/img_threshold.png", img)
    return img


def contours_text(orig, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        # rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Cropping the text block for giving input to OCR
        cropped = orig[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        config = '-l eng --oem 1 --psm 3'
        text = pytesseract.image_to_string(cropped, config=config)
        return text


def process_text(text: str):
    if text == '':
        return False
    else:
        splits = text.split(' ')
        splits = [x.strip() for x in splits]
        splits = [''.join(filter(lambda x: x.isdigit(), test_string)) for test_string in splits]
        flag = False
        for split in splits:
            if split.isnumeric():
                flag = True
                break
        if flag:
            print(splits)
        return flag


def process_frame(image):
    image_gray = gray(image)
    contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    text = contours_text(image, contours)
    return process_text(text)


def process_match(input_path, output_path):
    # Path to video file
    cap = cv2.VideoCapture(input_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_width, frame_height, frame_fps, frame_count)

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(output_path, fourcc, frame_fps, (frame_width, frame_height))
    cropped_height = int(0.8 * frame_height)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    while success:
        print("Processing frame number - {}".format(count))
        success, image = cap.read()
        if success:
            #cv2.imwrite("output_frames/highlight/frame_{}.jpg".format(count), image)
            cropped_image = image[cropped_height:frame_height, 0: frame_width]
            #cv2.imwrite("output_cropped_frames/highlight/frame_{}.jpg".format(count), cropped_image)
            # Writing the cropped frames
            if process_frame(cropped_image):
                out.write(image)
        count += 1
    print("Count of frames in original video - {}".format(count))
    cv2.destroyAllWindows()
    cap.release()
    out.release()


# Driver Code
if __name__ == '__main__':
    # Calling the function
    process_match("./input/match.mp4", "./output/match/match_full_highlights.mp4")