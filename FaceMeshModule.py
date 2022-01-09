import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, max_num_faces=2, static_image_mode=False, refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.max_num_faces = max_num_faces
        self.static_image_mode = static_image_mode
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            max_num_faces=self.max_num_faces, static_image_mode=self.static_image_mode,
            refine_landmarks=self.refine_landmarks, min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def findMeshFace(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if draw:
            if self.results.multi_face_landmarks:
                for faceId, faceLms in enumerate(self.results.multi_face_landmarks):  # faceLms = faceLandmarks
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=self.drawSpec,
                                               connection_drawing_spec=self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLms.landmark):
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        face.append([x, y])
                    faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findMeshFace(img)
        if len(faces) != 0:
            for face in faces:
                print(face)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
