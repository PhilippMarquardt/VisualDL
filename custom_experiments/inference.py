from visualdl import vdl
import cv2
import os
import sys


def main(model_path, folder_path, save_folder):
    if not os.path.exists(save_folder):
        print("SAVE FOLDER DOES NOT EXISTS CREATING...")
        os.mkdir(save_folder)
    print("LOADING MODEL")
    inf2 = vdl.get_inference_model(model_path)
    val_folder = folder_path
    print("PROCESSING IMAGES")
    for c, im in enumerate(os.listdir(val_folder)):
        img = cv2.imread(os.path.join(val_folder, im))[:, :, ::-1]

        img = cv2.resize(img, (512, 512))
        orig = img.copy()
        maps = inf2.predict([img])[0][0]

        liquid = (maps == 1).astype("uint8")
        bubble = (maps == 2).astype("uint8")

        contours_liquid, hierarchy = cv2.findContours(
            liquid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        contours_bubble, hierarchy = cv2.findContours(
            bubble, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        overlay = img.copy()
        cv2.fillPoly(overlay, contours_liquid, (255, 0, 0))
        cv2.fillPoly(overlay, contours_bubble, (0, 0, 255))
        cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0, img)
        cv2.imwrite(os.path.join(save_folder, f"overlayed{c}.png"), img)
        cv2.imwrite(os.path.join(save_folder, f"original{c}.png"), orig)

        cv2.imshow("overlayed", img)
        cv2.imshow("original", orig)
        cv2.waitKey()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
