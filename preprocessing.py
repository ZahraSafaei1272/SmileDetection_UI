import cv2
import glob
import os



def crop_image(image_paths):
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    crop_img = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = face_classifier.detectMultiScale(
                gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            if len(face) > 0:
                largest_face = max(face, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                croped_img = img[y:y+h, x:x+w]
                # Draw the merged box
                #rect_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                crop_img.append(croped_img)    
                #cv2.imshow('faces', img)
                #cv2.waitKey(200)
                #cv2.destroyAllWindows()
            else:
                crop_img.append(img)
        else:
            print(f"Failed to load: {path}")
    return crop_img    





def save_images(images, filename):
    output_dir = filename
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(images):
        file_name = os.path.join(output_dir, f'image_{idx+1}.jpg')
        success = cv2.imwrite(file_name, img)
        #if success:
            #print(f"Saved: {filename}")
        #else:
            #print(f"Failed to save: {filename}")
    




def resize_crop_img(images, target_w, target_h):
    resized_img = []
    for img in images:
        h, w = img.shape[:2]

        # Scale to fit target size while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create new image and paste resized in the center
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]  # black padding
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        resized_img.append(padded)
    return resized_img            



def main():
    image_paths = glob.glob('files/*.jpg')
    crop_images = crop_image(image_paths)
    save_images(crop_images, filename='cropped_images')
    target_w, target_h = (256,256)
    resized_images = resize_crop_img(crop_images, target_w, target_h)
    save_images(resized_images, filename='processed_images')



if __name__ == '__main__':
    main()






