import cv2

def stitch(frame, graph_img):
    h = max(frame.shape[0], graph_img.shape[0])

    frame = cv2.resize(frame, (640, h))
    graph_img = cv2.resize(graph_img, (640, h))

    return np.concatenate([frame, graph_img], axis=1)

class VideoWriter:
    def __init__(self, path, fps=10, frame_size=(1280, 720)):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(path, fourcc, fps, frame_size)

    def write(self, frame):
        self.out.write(frame)

    def close(self):
        self.out.release()