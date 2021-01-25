import cv2

def visualize_generator(gen):
    #gen = time_series_generator(64, train_dir)
    videos, labels = next(gen)
    for idx, video in enumerate(videos):
        label = labels[idx]
        display = True
        for f in video:
            # Display the resulting frame
            f = cv2.resize(f, (512, 512)).astype("float32")
            text = "activity: {}".format(label)
            cv2.putText(f, text, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
            cv2.imshow('Frame', f[:,:,::-1])
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                display = False
                break
        if not display:
            break
    cv2.destroyAllWindows()
    print(videos.shape)
    print(labels.shape)
