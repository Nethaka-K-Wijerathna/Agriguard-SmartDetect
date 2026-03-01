from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

image = cv2.imread('images.webp')

# detect objects in the image and show the results in imshow
results = model(image)

for r in results:
    im_array = r.plot()
    cv2.imshow('image', im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('image_result.jpg', im_array)