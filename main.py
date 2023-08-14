import PIL
import face_recognition
import os
import shutil
from tqdm import tqdm


def load_known_faces(faces_folder):
    # Load known faces
    known_faces = []
    for filename in tqdm(os.listdir(faces_folder), total=len(os.listdir(faces_folder)), desc="Loading known faces"):
        filepath = os.path.join(faces_folder, filename)
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=0)
        face_encodings = face_recognition.face_encodings(image, face_locations, model="large")
        if len(face_encodings) > 0:
            known_faces.extend(face_encodings)
        else:
            print(f"Warning: No faces found in {filepath}")

    print(f"Scanned {len(known_faces)} known faces")
    return known_faces


def create_output_folder(output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # ask user if they want to delete the output folder
        print(f"Warning: {output_folder} already exists. Do you want to delete it? (y/n)")
        answer = input()
        if answer == "y":
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
        else:
            print("Output folder not deleted. Images may be overwritten.")


def does_image_have_known_face(image_path, known_faces):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=0)
    face_encodings = face_recognition.face_encodings(image, face_locations, model="large")

    contains_known_face = False
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        if True in matches:
            contains_known_face = True
            break

    # release memory
    del image, face_locations, face_encodings

    return contains_known_face

def find_known_faces(faces_folder, pictures_folder, output_folder):
    known_faces = load_known_faces(faces_folder)

    if len(known_faces) == 0:
        print("No known faces found")
        return

    create_output_folder(output_folder)

    # Find and copy pictures with known faces
    total = 0
    for dirpath, dirnames, filenames in os.walk(pictures_folder):
        # print the directory we are in

        for filename in tqdm(filenames, total=len(filenames), desc=f"Scanning {dirpath}"):
            filepath = os.path.join(dirpath, filename)

            try:
                if does_image_have_known_face(filepath, known_faces):
                    # print(f"\nFound known face in {filepath}")

                    shutil.copy2(filepath, output_folder)
                    total += 1
                    continue
            except PIL.UnidentifiedImageError as e:
                continue
            except Exception as e:
                print(f"\nError: {e}")
                # move file to error folder
                error_folder = os.path.join(output_folder, "error")
                if not os.path.exists(error_folder):
                    os.makedirs(error_folder)
                shutil.copy2(filepath, error_folder)
                continue

    print(f"Found {total} pictures with known faces and copied them to {output_folder}")


def check_cuda():
    import dlib
    cuda_support = dlib.DLIB_USE_CUDA
    if not cuda_support:
        print("Warning: CUDA not supported. It is recommended to install CUDA and CUDNN for faster face recognition. "
              "Otherwise, face recognition will be too slow to be usable.")


if __name__ == "__main__":
    check_cuda()
    # Find known faces in pictures
    find_known_faces("known_faces", "pictures", "output")
